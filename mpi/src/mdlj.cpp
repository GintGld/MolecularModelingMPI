#include "mdlj.h"
#include "mdlj-mpi.h"

using std::cout;
using std::vector;
using std::to_string;
using std::ofstream;
using std::string;

__system_options OPTIONS;
vector<Particle> particles;
double potential_energy;
double kinetic_energy;

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
    cout << "\t -ecorr             Use energy correction" << "\n";
    cout << "\t -seed [integer]    Random number generator seed" << "\n";
    cout << "\t -novelo            Not print and not read velocity from files" << "\n";
    cout << "\t -h                 Print this info" << "\n";
}

void ParseCommandLineArguments(int argc, char** argv) {
    for (int arg_index = 1; arg_index < argc; ++arg_index) {
        string arg_str = argv[arg_index];
        if (arg_str == "-dx") OPTIONS.dimx = atoi(argv[++arg_index]);
        else if (arg_str == "-dy") OPTIONS.dimy = atoi(argv[++arg_index]);
        else if (arg_str == "-dz") OPTIONS.dimz = atoi(argv[++arg_index]);
        else if (arg_str == "-N") OPTIONS.global_particles_number = atoi(argv[++arg_index]);
        else if (arg_str == "-rho") OPTIONS.density = atof(argv[++arg_index]);
        else if (arg_str == "-dt") OPTIONS.dt = atof(argv[++arg_index]);
        else if (arg_str == "-rc") OPTIONS.cutoff_radius = atof(argv[++arg_index]);
        else if (arg_str == "-ns") OPTIONS.steps_number = atoi(argv[++arg_index]);
        else if (arg_str == "-T0") OPTIONS.T0 = atof(argv[++arg_index]);
        else if (arg_str == "-thermof") OPTIONS.print_thermo_frequency = atoi(argv[++arg_index]);
        else if (arg_str == "-outf") OPTIONS.print_out_frequency = atoi(argv[++arg_index]);
        else if (arg_str == "-ecorr") OPTIONS.use_energy_correction = true;
        else if (arg_str == "-seed") OPTIONS.seed = (unsigned)atoi(argv[++arg_index]);
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
        OPTIONS.global_particles_number
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

vector< vector<Particle> > __generate_positions_per_cell() {
    // Find the lowest perfect cube, n3, greater than or equal to the number of particles
    int lattice_size = 1;
    while (lattice_size * lattice_size * lattice_size 
            < (int)(OPTIONS.global_particles_number)) {
        lattice_size++;
    }

    vector< vector<Particle> > particles_divided_by_cells(MPI_OPTIONS.cells);

    // Generate particles in simple cubic (sc) lattice
    int index_x = 0, index_y = 0, index_z = 0, rank, coords[3] = {0,0,0};
    for (unsigned index = 0; index < OPTIONS.global_particles_number; ++index) {
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
        // coordinates are relative
        // (each cell has its own coordinate system)
        particles_divided_by_cells[rank].push_back({
            ((double)index_x + 0.5) * OPTIONS.simple_box_size * OPTIONS.dimx / lattice_size - coords[0] * OPTIONS.simple_box_size,
            ((double)index_y + 0.5) * OPTIONS.simple_box_size * OPTIONS.dimy / lattice_size - coords[1] * OPTIONS.simple_box_size,
            ((double)index_z + 0.5) * OPTIONS.simple_box_size * OPTIONS.dimz / lattice_size - coords[2] * OPTIONS.simple_box_size
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

void __generate_velocities(vector<vector<Particle>>& particles_per_cell) {
    std::normal_distribution<double> distribution(0, 1);
    
    std::mt19937 rng;

    for (auto& cell : particles_per_cell) {
        for (auto& p : cell) {
            p.vx = distribution(rng);
            p.vy = distribution(rng);
            p.vz = distribution(rng);
        }
    }

    // Get the velocity of system's center-of-mass
    Particle center_of_mass;  // Mind that the default vx, vy, vz for struct Particle is set to 0
    for (const auto& cell : particles_per_cell) {
        for (const auto& p : cell) {
            center_of_mass.vx += p.vx;
            center_of_mass.vy += p.vy;
            center_of_mass.vz += p.vz;
        }
    }
    // cout << MPI_OPTIONS.rank << ' ' << center_of_mass.vx << ' ' << center_of_mass.vy << ' ' << center_of_mass.vz << "\n";
    center_of_mass.vx /= OPTIONS.global_particles_number;
    center_of_mass.vy /= OPTIONS.global_particles_number;
    center_of_mass.vz /= OPTIONS.global_particles_number;
    
    // Take away any center-of-mass drift and calculate kinetic energy
    double kinetic_energy = 0.0;
    for (auto& cell : particles_per_cell) {
        for (auto& p : cell) {
            p.vx -= center_of_mass.vx;
            p.vy -= center_of_mass.vy;
            p.vz -= center_of_mass.vz;
            kinetic_energy += (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz) / 2;
        }
    }

    // Set the system's temperature to the initial temperature T0
    double T_current = kinetic_energy / OPTIONS.global_particles_number * 2 / 3;
    double velocity_factor = sqrt(OPTIONS.T0 / T_current);
    for (auto& cell : particles_per_cell) {
        for (auto& p : cell) {
            p.vx *= velocity_factor;
            p.vy *= velocity_factor;
            p.vz *= velocity_factor;
        }
    }

    // cout << MPI_OPTIONS.rank;
    // for (const auto& p : particles)
    //     cout << p.vx << ' ' << p.vy << ' ' << p.vz << "\n";
}

void __write_particles_XYZ(ofstream& stream, Particle* buff, double time) {
    stream << OPTIONS.global_particles_number << "\n";
    stream << "Lattice=\" " 
        << OPTIONS.simple_box_size * OPTIONS.dimx << " 0.0 0.0 0.0 "
        << OPTIONS.simple_box_size * OPTIONS.dimy << " 0.0 0.0 0.0 " 
        << OPTIONS.simple_box_size * OPTIONS.dimz << " \"";
    if (OPTIONS.read_and_print_with_velocity) {
        stream << " Properties=pos:R:3:velo:R:3";
    }
    else {
        stream << " Properties=pos:R:3";
    }
    stream << " Time = " << time << "\n";
    for (unsigned i = 0; i < OPTIONS.global_particles_number; ++i)
        stream << buff[i].x << ' ' << buff[i].y << ' ' << buff[i].z << ' '
               << buff[i].vx << ' ' << buff[i].vy << ' ' << buff[i].vz << "\n";
}

void __first_half_step() {
    for (auto& p : particles) {
        p.x += p.vx * OPTIONS.dt + 0.5 * OPTIONS.dt * OPTIONS.dt * p.ax;
        p.y += p.vy * OPTIONS.dt + 0.5 * OPTIONS.dt * OPTIONS.dt * p.ay;
        p.z += p.vz * OPTIONS.dt + 0.5 * OPTIONS.dt * OPTIONS.dt * p.az;

        p.vx += 0.5 * OPTIONS.dt * p.ax;
        p.vy += 0.5 * OPTIONS.dt * p.ay;
        p.vz += 0.5 * OPTIONS.dt * p.az;
    }
}

void __second_half_step() {
    kinetic_energy = 0.0;
    for (auto& p : particles) {
        p.vx += 0.5 * OPTIONS.dt * p.ax;
        p.vy += 0.5 * OPTIONS.dt * p.ay;
        p.vz += 0.5 * OPTIONS.dt * p.az;

        kinetic_energy += (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz) / 2;
    }
}