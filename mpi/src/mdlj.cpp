#include "mdlj.h"
#include "mdlj-mpi.h"

using std::cout;
using std::vector;
using std::to_string;
using std::ofstream;
using std::string;

__system_options OPTIONS;
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
            cout << "Error: Command-line argument '" << arg_str << "' not recognized." << "\n";
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

vector< vector<Particle> > __GeneratePositionsPerCell() {
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

    return particles_divided_by_cells;
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

void __WriteParticlesXYZ(ofstream& stream, Particle* buff) {
    /*
     * function for testing WriteParticlesXYZ_MPI
    */

    for (int i = 0; i < OPTIONS.particles_number; ++i)
        cout << buff[i].x << ' ' << buff[i].y << ' ' << buff[i].z << ' '
             << buff[i].vx << ' ' << buff[i].vy << ' ' << buff[i].vz << "\n";
}