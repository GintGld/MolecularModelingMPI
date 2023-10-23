#define _USE_MATH_DEFINES
#define ROOT_PROCESS 0

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <mpi.h>

// writing in one line causes c++17 warnings
using std::cout;
using std::cerr;
using std::to_string;
using std::string;
using std::vector;
using std::ofstream;

struct Particle {
    double x, y, z;
    double vx = 0, vy = 0, vz = 0;
    double ax = 0, ay = 0, az = 0;
    double image_x = 0, image_y = 0, image_z = 0;
};

struct __system_options {
    int dimx, dimy, dimz;
    unsigned particles_number = 216;
    double density = 0.5;
    double simple_box_size;
    double T0 = 1.0;
    unsigned seed = time(NULL);
    double cutoff_radius = 1.0e6;
    double energy_cut;
    double energy_correction = 0.0;
    double dt = 0.001;
    unsigned steps_number = 100;
    unsigned print_thermo_frequency = 100;
    unsigned print_out_frequency = 100;
    bool write_output_in_one_file = false;
    bool use_energy_correction = false;
    bool print_unfolded_coordinates = false;
    bool read_and_print_with_velocity = true;
};

struct __mpi_options {
    MPI_Datatype dt_particles;
    MPI_Comm COMM = MPI_COMM_WORLD;
    int cells, rank, coords[3], neighbors[3][2];
};

// global values for program configurations
MPI_Comm COMM;
__system_options OPTIONS;
__mpi_options MPI_OPTIONS;
vector<Particle> particles;

void PrintUsageInfo() {
    cout << "mdlj usage:" << "\n";
    cout << "mdlj [options]" << "\n\n";
    cout << "Options:" << "\n";
    cout << "\t -dx [integer]      dimx for MPI\n";
    cout << "\t -dy [integer]      dimy for MPI\n";
    cout << "\t -dz [integer]      dimz for MPI\n";
    cout << "\t -N [integer]       Number of particles" << "\n";
    cout << "\t -rho [real]        Number density" << "\n";
    cout << "\t -dt [real]         Time step" << "\n";
    cout << "\t -rc [real]         Cutoff radius" << "\n";
    cout << "\t -ns [real]         Number of integration steps" << "\n";
    cout << "\t -T0 [real]         Initial temperature" << "\n";
    cout << "\t -thermof [integer] Thermo information print frequency" << "\n";
    cout << "\t -outf [integer]    Print positions to XYZ file frequency" << "\n";
    cout << "\t -onefile           Write config to single output file (multiple files otherwise)"
        << "\n";
    cout << "\t -ecorr             Use energy correction" << "\n";
    cout << "\t -seed [integer]    Random number generator seed" << "\n";
    cout << "\t -uf                Print unfolded coordinates in output files" << "\n";
    cout << "\t -novelo            Not print and not read velocity from files" << "\n";
    cout << "\t -h                 Print this info" << "\n";
}

void ParseCommandLineArguments(int argc, char** argv) {
    for (int arg_index = 1; arg_index < argc; ++arg_index) {
        string arg_str = argv[arg_index];
        if (arg_str == "-dx") OPTIONS.dimx = atoi(argv[++arg_index]);
        else if (arg_str == "-dy") OPTIONS.dimy = atoi(argv[++arg_index]);
        else if (arg_str == "-dz") OPTIONS.dimz = atoi(argv[++arg_index]);
        else if (arg_str == "-N") OPTIONS.particles_number = atoi(argv[++arg_index]);
        else if (arg_str == "-rho") OPTIONS.density = atof(argv[++arg_index]);
        else if (arg_str == "-dt") OPTIONS.dt = atof(argv[++arg_index]);
        else if (arg_str == "-rc") OPTIONS.cutoff_radius = atof(argv[++arg_index]);
        else if (arg_str == "-ns") OPTIONS.steps_number = atoi(argv[++arg_index]);
        else if (arg_str == "-T0") OPTIONS.T0 = atof(argv[++arg_index]);
        else if (arg_str == "-thermof") OPTIONS.print_thermo_frequency = atoi(argv[++arg_index]);
        else if (arg_str == "-outf") OPTIONS.print_out_frequency = atoi(argv[++arg_index]);
        else if (arg_str == "-onefile") OPTIONS.write_output_in_one_file = true;
        else if (arg_str == "-ecorr") OPTIONS.use_energy_correction = true;
        else if (arg_str == "-seed") OPTIONS.seed = (unsigned)atoi(argv[++arg_index]);
        else if (arg_str == "-uf") OPTIONS.print_unfolded_coordinates = true;
        else if (arg_str == "-novelo") OPTIONS.read_and_print_with_velocity = false;
        else if (arg_str == "-h") {
            PrintUsageInfo();
            MPI_Abort(COMM, EXIT_SUCCESS);
        }
        else {
            cerr << "Error: Command-line argument '" << arg_str << "' not recognized." << "\n";
            MPI_Abort(COMM, EXIT_FAILURE);
        }
    }

    OPTIONS.simple_box_size = cbrt(
        OPTIONS.particles_number
        / OPTIONS.density
        / OPTIONS.dimx
        / OPTIONS.dimy
        / OPTIONS.dimz
    );
    double rr3 = 1 / (OPTIONS.cutoff_radius * OPTIONS.cutoff_radius * OPTIONS.cutoff_radius);
    OPTIONS.energy_cut = 4 * (rr3 * rr3 * rr3 * rr3 - rr3 * rr3);
    if (OPTIONS.use_energy_correction) {
        OPTIONS.energy_correction = 8 * M_PI * OPTIONS.density * (rr3 * rr3 * rr3 / 9.0 - rr3 / 3.0);
    }
}

void MPI_Apply(int ierr, string error_message) {
    /*
     * Wrapper for all MPI functions that
     * return error status as output
     * (almost all according documentation)
    */
    if (ierr != MPI_SUCCESS) {
        cout << "USER CATCHED ERROR [" << ierr << "]\n"
             << error_message << "\n";
        MPI_Abort(COMM, EXIT_FAILURE);
    }
}

void init_cart_comm() {
    /*
     * create Cartesian communicator and
     * initiate `MPIOptions` member that will save
     * all information about current process configuration
    */

    // check if number of workers equals number of cells
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_OPTIONS.cells);
    int cells_theory = OPTIONS.dimx * OPTIONS.dimy * OPTIONS.dimz;
    if (MPI_OPTIONS.cells != cells_theory) {
        cout << "ERROR: number of processes not equal to number of cells.\n"
            << "Processes:\t" << MPI_OPTIONS.cells << "\n"
            << "Cells:\t\t" << cells_theory << "\n";
        MPI_Abort(COMM, EXIT_FAILURE);
    }

    // init particle data type
    MPI_Apply(
        MPI_Type_contiguous(12, MPI_DOUBLE, &MPI_OPTIONS.dt_particles),
        "Fail in creating particle mpi-data type"
    );
    MPI_Apply(
        MPI_Type_commit(&MPI_OPTIONS.dt_particles),
        "Fail in commiting particle mpi-data type"
    );

    // Create Cartesian Commutator
    int reorder = 0;
    int dims[3] = {OPTIONS.dimx, OPTIONS.dimy, OPTIONS.dimz}, periods[3] = {1,1,1};
    MPI_Apply(
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &COMM),
        "Fail to create Cartesian communicator"
    );
    
    // Get rank of the process
    MPI_Apply(
        MPI_Cart_map(COMM, 3, dims, periods, &MPI_OPTIONS.rank),
        string("Fail in getting rank of the process.\n")
    );

    // Get process coordinates
    MPI_Apply(
        MPI_Cart_coords(COMM, MPI_OPTIONS.rank, 3, MPI_OPTIONS.coords),
        string("fail in getting coordinates.\n") +
        string("rank:\t") + to_string(MPI_OPTIONS.rank)
    );
}

void GeneratePositions_MPI() {
    /*
     * Generates all particles in root process memory
     * then send it to all other processes
    */
    
    int number_of_particles;
    
    // pointers for `MPI_Scatterv`
    Particle *sendbuff = nullptr, *particles_for_current_process = nullptr;
    int *counts = nullptr, *displs = nullptr;

    // root process generates all particles in a sc1 and 
    // scatters particles for their cells
    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        // Find the lowest perfect cube, n3, greater than or equal to the number of particles
        int lattice_size = 1;
        while (lattice_size * lattice_size * lattice_size 
               * OPTIONS.dimx * OPTIONS.dimy * OPTIONS.dimz 
               < (int)(OPTIONS.particles_number)) {
            lattice_size++;
        }

        vector< vector<Particle> > particles_divided_by_cells(MPI_OPTIONS.cells);

        // Generate particles in simple cubic (sc) lattice
        int index_x = 0, index_y = 0, index_z = 0, rank, coords[3] = {0,0,0};
        for (unsigned index = 0; index < OPTIONS.particles_number; ++index) {
            // get coordinates for corresponding cell
            coords[0] = (index_x * OPTIONS.dimx) / lattice_size;
            coords[1] = (index_y * OPTIONS.dimy) / lattice_size;
            coords[2] = (index_z * OPTIONS.dimz) / lattice_size;
            MPI_Apply(
                MPI_Cart_rank(COMM, coords, &rank),
                string("Fail in getting rank from coordinates\n") + 
                "coordinates\t" + to_string(coords[0]) + ' ' + to_string(coords[1]) + ' ' + to_string(coords[2]) + 
                "process rank\t" + to_string(MPI_OPTIONS.rank)
            );

            // push particle for corresponding vector
            particles_divided_by_cells[rank].push_back({
                ((double)index_x + 0.5) * OPTIONS.simple_box_size * OPTIONS.dimx / lattice_size,
                ((double)index_y + 0.5) * OPTIONS.simple_box_size * OPTIONS.dimy / lattice_size,
                ((double)index_z + 0.5) * OPTIONS.simple_box_size * OPTIONS.dimz / lattice_size
            });

            // make step for geometric indices
            index_x++;
            if (index_x == lattice_size) {
                index_x = 0;
                index_y++;
                if (index_y == lattice_size) {
                    index_y = 0;
                    index_z++;
                }
            }
        }


        // Construct info arrays for `MPI_Scatterv` from vector
        counts = new int[MPI_OPTIONS.cells];
        displs = new int[MPI_OPTIONS.cells];
        for (int i = 0; i < MPI_OPTIONS.cells; ++i) {
            counts[i] = particles_divided_by_cells[i].size();
            displs[i] = (i == 0)? 0 : displs[i - 1] + counts[i - 1];
        }

        // flatten particles
        sendbuff = new Particle[OPTIONS.particles_number];
        for (int i = 0; i < MPI_OPTIONS.cells; ++i) {
            for (int j = 0; j < counts[i]; ++j) {
                sendbuff[displs[i] + j] = particles_divided_by_cells[i][j];
           
            }
        }
    }

    // send number of particles for each cell
    MPI_Apply(
       MPI_Scatter(counts, 1, MPI_INT,
                   &number_of_particles, 1, MPI_INT, 
                   ROOT_PROCESS, COMM),
       string("Fail in GeneratePositions_MPI -> MPI_Scatter\n") + 
       "process rank\t" + to_string(MPI_OPTIONS.rank)
    );
    
    // allocate memory for input
    particles_for_current_process = new Particle[number_of_particles];

    // send particles for cells
    MPI_Apply(
        MPI_Scatterv(sendbuff, counts, displs, MPI_OPTIONS.dt_particles,
                     particles_for_current_process, number_of_particles,
                     MPI_OPTIONS.dt_particles, ROOT_PROCESS, COMM),
        string("Fail in GeneratePositions_MPI -> MPI_Scatter\n") + 
        "process rank\t" + to_string(MPI_OPTIONS.rank)
    );

    // convert to vector
    particles.clear();
    for (int i = 0; i < number_of_particles; ++i)
        particles.push_back(particles_for_current_process[i]);

    delete[] particles_for_current_process;
    delete[] sendbuff;
    delete[] counts;
    delete[] displs;
}

void WriteParticlesXYZ(ofstream& stream, Particle* buff) {
    /*
     * function for testing WriteParticlesXYZ_MPI
    */

    for (int i = 0; i < OPTIONS.particles_number; ++i)
        cout << buff[i].x << ' ' << buff[i].y << ' ' << buff[i].z << ' '
             << buff[i].vx << ' ' << buff[i].vy << ' ' << buff[i].vz << "\n";
}

void WriteParticlesXYZ_MPI(ofstream& stream) {
    
    // Pointers for `MPI_Gatherv`, neccecary only in root process
    Particle *GatheredParticles = nullptr;
    int *counts = nullptr, *displs = nullptr;

    // allocate memory for `counts` array
    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        counts = new int[MPI_OPTIONS.cells];
    }

    // get number of particles in each cell
    int particles_per_cell = particles.size();
    MPI_Apply(
        MPI_Gather(&particles_per_cell, 1, MPI_INT,
                   counts, 1, MPI_INT,
                   ROOT_PROCESS, COMM),
        string("Fail in WriteParticlesXYZ_MPI -> MPI_Gather\n") + 
        "process rank\t" + to_string(MPI_OPTIONS.rank)
    );

    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        // allocate memory for `displs` array
        displs = new int[MPI_OPTIONS.cells];
        for(int i = 0; i < MPI_OPTIONS.cells; ++i)
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        
        // allocate memory for `recvbuf` array
        GatheredParticles = new Particle[OPTIONS.particles_number];
    }

    // Get particles from all cells
    MPI_Apply(
        MPI_Gatherv(particles.data(), particles.size(), MPI_OPTIONS.dt_particles,
                    GatheredParticles, counts, displs, MPI_OPTIONS.dt_particles,
                    ROOT_PROCESS, COMM),
        string("Fail in WriteParticlesXYZ_MPI -> MPI_Gatherv\n") + 
        "process rank\t" + to_string(MPI_OPTIONS.rank)
    );

    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        WriteParticlesXYZ(stream, GatheredParticles);
    }

    delete[] counts;
    delete[] displs;
    delete[] GatheredParticles;
}

void GenerateVelocities(std::mt19937& rng) {
    std::normal_distribution<double> distribution(0, 1);
    
    for (auto& particle : particles) {
        particle.vx = distribution(rng);
        particle.vy = distribution(rng);
        particle.vz = distribution(rng);
    }

    // Get the velocity of system's center-of-mass
    Particle center_of_mass;  // Mind that the default vx, vy, vz for struct Particle is set to 0
    for (const auto& particle : particles) {
        center_of_mass.vx += particle.vx;
        center_of_mass.vy += particle.vy;
        center_of_mass.vz += particle.vz;
    }
    center_of_mass.vx /= OPTIONS.particles_number;
    center_of_mass.vy /= OPTIONS.particles_number;
    center_of_mass.vz /= OPTIONS.particles_number;
    
    // Take away any center-of-mass drift and calculate kinetic energy
    double kinetic_energy = 0.0;
    for (auto& particle : particles) {
        particle.vx -= center_of_mass.vx;
        particle.vy -= center_of_mass.vy;
        particle.vz -= center_of_mass.vz;
        kinetic_energy += (particle.vx * particle.vx 
            + particle.vy * particle.vy 
            + particle.vz * particle.vz) / 2;
    }

    // Set the system's temperature to the initial temperature T0
    double T_current = kinetic_energy / OPTIONS.particles_number * 2 / 3;
    double velocity_factor = sqrt(OPTIONS.T0 / T_current);
    for (auto& particle : particles) {
        particle.vx *= velocity_factor;
        particle.vy *= velocity_factor;
        particle.vz *= velocity_factor;
    }
}

// write function to get energy

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    ParseCommandLineArguments(argc, argv);

    init_cart_comm();

    // Output some initial information
    // if (MPI_OPTIONS.rank == ROOT_PROCESS) {
    //     cout << "# NVE MD Simulation of a Lennard - Jones fluid" << "\n";
    //     cout << "# dims = [" << OPTIONS.dimx << ", " << OPTIONS.dimy << ", " << OPTIONS.dimz << "]\n";
    //     cout << "# L = " << OPTIONS.simple_box_size << " rho = " << OPTIONS.density << " N = "
    //         << OPTIONS.particles_number << " r_cut = " << OPTIONS.cutoff_radius << "\n";
    //     cout << "# Steps number = " << OPTIONS.steps_number << " seed = " << OPTIONS.seed 
    //         << " dt = " << OPTIONS.dt << "\n";
    // }

    GeneratePositions_MPI();

    // cout << MPI_OPTIONS.rank << "\n";
    // for (const auto& p : particles)
    //     cout << p.x << ' ' << p.y << ' ' << p.z << "\n";
    // cout << "\n";

    std::mt19937 rng;

    GenerateVelocities(rng);


    ofstream out;
    WriteParticlesXYZ_MPI(out);

    // Free the datatype
    MPI_Type_free(&MPI_OPTIONS.dt_particles);
 
    // Finish with mpi
    MPI_Finalize();


}
