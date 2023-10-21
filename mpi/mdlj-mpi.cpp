/*
   Code for Microcanonical Molecular Dynamics simulation of a Lennard-Jones
   system in a periodic boundary

   Written by Vladislav Negodin

   Based.
   And based on C code by Cameron F. Abrams, 2004

   compile using "g++ -o mdlj mdlj.cpp"
*/

#define _USE_MATH_DEFINES
#define ROOT_PROCESS 0

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <random>
#include <ctime>
#include <cmath>
#include <exception>
#include <mpi.h>

// writing in one line causes c++17 warnings
using std::cout;
using std::cerr;
using std::to_string;
using std::string;
using std::vector;
using std::ofstream;

// global communicator
MPI_Comm COMM;

struct Particle {
    double x, y, z;
    double vx = 0, vy = 0, vz = 0;
    double ax = 0, ay = 0, az = 0;
    double image_x = 0, image_y = 0, image_z = 0;
};

struct SystemOptions {
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

struct MPIOptions {
    MPI_Datatype dt_particles;
    MPI_Comm COMM = MPI_COMM_WORLD;
    int cells, rank, coords[3], neighbors[3][2];
};

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

SystemOptions ParseCommandLineArguments(int argc, char** argv) {
    SystemOptions options;
    for (int arg_index = 1; arg_index < argc; ++arg_index) {
        string arg_str = argv[arg_index];
        if (arg_str == "-dx") options.dimx = atoi(argv[++arg_index]);
        else if (arg_str == "-dy") options.dimy = atoi(argv[++arg_index]);
        else if (arg_str == "-dz") options.dimz = atoi(argv[++arg_index]);
        else if (arg_str == "-N") options.particles_number = atoi(argv[++arg_index]);
        else if (arg_str == "-rho") options.density = atof(argv[++arg_index]);
        else if (arg_str == "-dt") options.dt = atof(argv[++arg_index]);
        else if (arg_str == "-rc") options.cutoff_radius = atof(argv[++arg_index]);
        else if (arg_str == "-ns") options.steps_number = atoi(argv[++arg_index]);
        else if (arg_str == "-T0") options.T0 = atof(argv[++arg_index]);
        else if (arg_str == "-thermof") options.print_thermo_frequency = atoi(argv[++arg_index]);
        else if (arg_str == "-outf") options.print_out_frequency = atoi(argv[++arg_index]);
        else if (arg_str == "-onefile") options.write_output_in_one_file = true;
        else if (arg_str == "-ecorr") options.use_energy_correction = true;
        else if (arg_str == "-seed") options.seed = (unsigned)atoi(argv[++arg_index]);
        else if (arg_str == "-uf") options.print_unfolded_coordinates = true;
        else if (arg_str == "-novelo") options.read_and_print_with_velocity = false;
        else if (arg_str == "-h") {
            PrintUsageInfo();
            MPI_Abort(COMM, EXIT_SUCCESS);
        }
        else {
            cerr << "Error: Command-line argument '" << arg_str << "' not recognized." << "\n";
            MPI_Abort(COMM, EXIT_FAILURE);
        }
    }

    options.simple_box_size = cbrt(
        options.particles_number
        / options.density
        / options.dimx
        / options.dimy
        / options.dimz
    );
    double rr3 = 1 / (options.cutoff_radius * options.cutoff_radius * options.cutoff_radius);
    options.energy_cut = 4 * (rr3 * rr3 * rr3 * rr3 - rr3 * rr3);
    if (options.use_energy_correction) {
        options.energy_correction = 8 * M_PI * options.density * (rr3 * rr3 * rr3 / 9.0 - rr3 / 3.0);
    }

    return options;
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

void init_cart_comm(const SystemOptions& options, MPIOptions& mpi_options) {
    /*
     * create Cartesian communicator and
     * initiate `MPIOptions` member that will save
     * all information about current process configuration
    */

    // check if number of workers equals number of cells
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_options.cells);
    int cells_theory = options.dimx * options.dimy * options.dimz;
    if (mpi_options.cells != cells_theory) {
        cout << "ERROR: number of processes not equal to number of cells.\n"
            << "Processes:\t" << mpi_options.cells << "\n"
            << "Cells:\t\t" << cells_theory << "\n";
        MPI_Abort(COMM, EXIT_FAILURE);
    }

    // init particle data type
    MPI_Apply(
        MPI_Type_contiguous(12, MPI_DOUBLE, &mpi_options.dt_particles),
        "Fail in creating particle mpi-data type"
    );
    MPI_Apply(
        MPI_Type_commit(&mpi_options.dt_particles),
        "Fail in commiting particle mpi-data type"
    );

    // Create Cartesian Commutator
    int reorder = 0;
    int dims[3] = {options.dimx, options.dimy, options.dimz}, periods[3] = {1,1,1};
    MPI_Apply(
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &COMM),
        "Fail to create Cartesian communicator"
    );
    
    // Get rank of the process
    MPI_Apply(
        MPI_Cart_map(COMM, 3, dims, periods, &mpi_options.rank),
        string("Fail in getting rank of the process.\n")
    );

    // Get process coordinates
    MPI_Apply(
        MPI_Cart_coords(COMM, mpi_options.rank, 3, mpi_options.coords),
        string("fail in getting coordinates.\n") +
        string("rank:\t") + to_string(mpi_options.rank)
    );
}

// rewrite in terms of GeneratePositions_MPI
void WriteParticlesXYZ_MPI(ofstream& streamer, const vector<Particle>& particles,
                           const SystemOptions& options, const MPIOptions& mpi_options) {
    
    // Define pointers for `MPI_Gatherv`, neccecary only in root process
    Particle *GatheredParticles = nullptr;
    int *counts = nullptr, *displs = nullptr;

    // Create array `counts` for `MPI_Gatherv`
    // will be fulfilled in MPI_Gather
    //if (mpi_options.rank == ROOT_PROCESS) {
        counts = new int[mpi_options.cells];
    //}

    // Gather numbers of particles from each process
    int particles_in_cell = (int)(particles.size());
    MPI_Apply(
        MPI_Gather(&particles_in_cell, 1, MPI_INT,
                   counts, mpi_options.cells, MPI_INT,
                   ROOT_PROCESS, COMM),
        string("Fail in WriteParticlesXYZ_MPI -> MPI_Gather\n") + 
        "Sending number of particles to root process\n" + 
        "process_rank\t" + to_string(mpi_options.rank)
    );

    cout << mpi_options.rank << ' ' << particles_in_cell << "\n";
    for (int i = 0; i < mpi_options.cells; ++i)
        cout << counts[i] << ' ';
    cout << "\n";

    delete[] GatheredParticles;
    delete[] counts;
    delete[] displs;

    return;

    // Create array `displs` for `MPI_Gatherv` (for root process)
    if (mpi_options.rank == ROOT_PROCESS) {
        displs = new int[mpi_options.cells];
        displs[0] = 0;
        for (int i = 1; i < mpi_options.cells; ++i) {
            displs[i] = displs[i - 1] + counts[i - 1];
        }
    }

    // Gather particles from all processes via `MPI_Gatherv`
    GatheredParticles = new Particle[options.particles_number];
    MPI_Apply(
        MPI_Gatherv(particles.data(), particles.size(), mpi_options.dt_particles,
                    GatheredParticles, counts, displs, mpi_options.dt_particles,
                    ROOT_PROCESS, COMM),
        string("Fail in WriteParticlesXYZ_MPI -> MPI_Gatherv\n") + 
        "Sending particles to root process\n" + 
        "process_rank\t" + to_string(mpi_options.rank)
    );

    delete[] GatheredParticles;
    delete[] counts;
    delete[] displs;

    return;
    // переписать https://rookiehpc.org/mpi/docs/mpi_in_place/index.html
    MPI_Apply(
        MPI_Gatherv(particles.data(), particles.size(), mpi_options.dt_particles,
                           GatheredParticles, counts, displs, mpi_options.dt_particles,
                           ROOT_PROCESS, COMM),
        string("Fail in WriteParticlesXYZ_MPI -> MPI_Gatherv.\n") + 
        "process_rank\t" + to_string(mpi_options.rank)
    );

    if (mpi_options.rank == ROOT_PROCESS) {
        int id_x = 0, id_y = 0, id_z = 0;
        int corner_rank, curr_rank, corner_coordinates[3] = {0, 0, 0};

        // get rank of process with (0,0,0) coordinates
        MPI_Apply(
            MPI_Cart_rank(COMM, corner_coordinates, &corner_rank),
            string("ERROR: fail in getting rank of the corner process (0,0,0).\n") + 
            "rank: " + to_string(mpi_options.rank)
        );

        for (unsigned i = 0; i < mpi_options.cells; ++i) {
            // get rank of the next process in geometric order
            curr_rank = corner_rank;
            MPI_Apply(
                MPI_Cart_shift(COMM, 0, id_x, &curr_rank, &curr_rank),
                string("ERROR: fail in getting shift in X dimension.\n") + 
                "rank: " + to_string(mpi_options.rank) + "\n" + 
                "tagret coords: " + to_string(id_x) + ' ' + 
                to_string(id_y) + ' ' + to_string(id_z)
            );
            MPI_Apply(
                MPI_Cart_shift(COMM, 1, id_y, &curr_rank, &curr_rank),
                string("ERROR: fail in getting shift in Y dimension.\n") + 
                "rank: " + to_string(mpi_options.rank) + "\n" + 
                "tagret coords: " + to_string(id_x) + ' ' + 
                to_string(id_y) + ' ' + to_string(id_z)
            );
            MPI_Apply(
                MPI_Cart_shift(COMM, 2, id_z, &curr_rank, &curr_rank),
                string("ERROR: fail in getting shift in Z dimension.\n") + 
                "rank: " + to_string(mpi_options.rank) + "\n" + 
                "tagret coords: " + to_string(id_x) + ' ' + 
                to_string(id_y) + ' ' + to_string(id_z)
            );
            
            // write particles
            for (int k = 0; k < counts[curr_rank]; ++k) {
                Particle p = GatheredParticles[k + displs[i]];
                streamer << p.x << ' ' << p.y << ' ' << p.z << ' ' << p.vx << ' ' << p.vy << ' ' << p.vz << "\n";
            }

            // make step of geometric indices
            id_x++;
            if (id_x == options.dimx) {
                id_x = 0;
                id_y++;
                if (id_y == options.dimy) {
                    id_y = 0;
                    id_z++;
                }
            }
        }
    }

    delete[] GatheredParticles; GatheredParticles = nullptr;
    delete[] counts; counts = nullptr;
    delete[] displs; displs = nullptr;
}


vector<Particle> GeneratePositions_MPI(const SystemOptions& options, const MPIOptions& mpi_options, MPI_Comm COMM) {
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
    if (mpi_options.rank == ROOT_PROCESS) {
        // Find the lowest perfect cube, n3, greater than or equal to the number of particles
        int lattice_size = 1;
        while (lattice_size * lattice_size * lattice_size 
               * options.dimx * options.dimy * options.dimz 
               < (int)(options.particles_number)) {
            lattice_size++;
        }

        vector< vector<Particle> > particles_divided_by_cells(mpi_options.cells);

        // Generate particles in simple cubic (sc) lattice
        int index_x = 0, index_y = 0, index_z = 0, rank, coords[3] = {0,0,0};
        for (unsigned index = 0; index < options.particles_number; ++index) {
            // get coordinates for corresponding cell
            coords[0] = (index_x * options.dimx) / lattice_size;
            coords[1] = (index_y * options.dimy) / lattice_size;
            coords[2] = (index_z * options.dimz) / lattice_size;
            MPI_Apply(
                MPI_Cart_rank(COMM, coords, &rank),
                string("Fail in getting rank from coordinates\n") + 
                "coordinates\t" + to_string(coords[0]) + ' ' + to_string(coords[1]) + ' ' + to_string(coords[2]) + 
                "process rank\t" + to_string(mpi_options.rank)
            );

            // push particle for corresponding vector
            particles_divided_by_cells[rank].push_back({
                ((double)index_x + 0.5) * options.simple_box_size * options.dimx / lattice_size,
                ((double)index_y + 0.5) * options.simple_box_size * options.dimy / lattice_size,
                ((double)index_z + 0.5) * options.simple_box_size * options.dimz / lattice_size
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
        counts = new int[mpi_options.cells];
        displs = new int[mpi_options.cells];
        for (int i = 0; i < mpi_options.cells; ++i) {
            counts[i] = particles_divided_by_cells[i].size();
            displs[i] = (i == 0)? 0 : displs[i - 1] + counts[i - 1];
        }

        // flatten particles
        sendbuff = new Particle[options.particles_number];
        for (int i = 0; i < mpi_options.cells; ++i) {
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
       "process rank\t" + to_string(mpi_options.rank)
    );
    
    // allocate memory for input
    particles_for_current_process = new Particle[number_of_particles];

    // send particles for cells
    MPI_Apply(
        MPI_Scatterv(sendbuff, counts, displs, mpi_options.dt_particles,
                     particles_for_current_process, number_of_particles,
                     mpi_options.dt_particles, ROOT_PROCESS, COMM),
        string("Fail in GeneratePositions_MPI -> MPI_Scatter\n") + 
        "process rank\t" + to_string(mpi_options.rank)
    );

    // convert to vector
    vector<Particle> particles;
    for (int i = 0; i < number_of_particles; ++i)
        particles.push_back(particles_for_current_process[i]);

    delete[] particles_for_current_process;
    delete[] sendbuff;
    delete[] counts;
    delete[] displs;
    
    return particles;
}

void GenerateVelocities(vector<Particle>& particles, const SystemOptions& options, std::mt19937& rng) {
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
    center_of_mass.vx /= options.particles_number;
    center_of_mass.vy /= options.particles_number;
    center_of_mass.vz /= options.particles_number;
    
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
    double T_current = kinetic_energy / options.particles_number * 2 / 3;
    double velocity_factor = sqrt(options.T0 / T_current);
    for (auto& particle : particles) {
        particle.vx *= velocity_factor;
        particle.vy *= velocity_factor;
        particle.vz *= velocity_factor;
    }
}

// write function to get energy

int main(int argc, char* argv[]) {
    SystemOptions options = ParseCommandLineArguments(argc, argv);

    MPI_Init(&argc, &argv);

    MPIOptions mpi_options;
    init_cart_comm(options, mpi_options);

    // Output some initial information
    // if (mpi_options.rank == ROOT_PROCESS) {
    //     cout << "# NVE MD Simulation of a Lennard - Jones fluid" << "\n";
    //     cout << "# dims = [" << options.dimx << ", " << options.dimy << ", " << options.dimz << "]\n";
    //     cout << "# L = " << options.simple_box_size << " rho = " << options.density << " N = "
    //         << options.particles_number << " r_cut = " << options.cutoff_radius << "\n";
    //     cout << "# Steps number = " << options.steps_number << " seed = " << options.seed 
    //         << " dt = " << options.dt << "\n";
    // }

    vector<Particle> particles = GeneratePositions_MPI(options, mpi_options, COMM);

    cout << mpi_options.rank << "\n";
    for (const auto& p : particles)
        cout << p.x << ' ' << p.y << ' ' << p.z << "\n";
    cout << "\n";

    // Free the datatype created
    MPI_Type_free(&mpi_options.dt_particles);
 
    // Finish with mpi
    MPI_Finalize();


}
