/*
   Code for Microcanonical Molecular Dynamics simulation of a Lennard-Jones
   system in a periodic boundary

   Written by Vladislav Negodin

   Based.
   And based on C code by Cameron F. Abrams, 2004

   compile using "g++ -o mdlj mdlj.cpp"
*/

#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <random>
#include <ctime>
#include <cmath>

using namespace std;

struct Particle {
    double x, y, z;
    double vx = 0, vy = 0, vz = 0;
    double ax = 0, ay = 0, az = 0;
    double image_x = 0, image_y = 0, image_z = 0;
};

struct SystemOptions {
    unsigned particles_number = 216;
    double density = 0.5;
    double box_size;
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
    string init_config_file = "";
    bool use_energy_correction = false;
    bool print_unfolded_coordinates = false;
    bool read_and_print_with_velocity = true;
};

void PrintUsageInfo() {
    cout << "mdlj usage:" << endl;
    cout << "mdlj [options]" << endl << endl;
    cout << "Options:" << endl;
    cout << "\t -N [integer]       Number of particles" << endl;
    cout << "\t -rho [real]        Number density" << endl;
    cout << "\t -dt [real]         Time step" << endl;
    cout << "\t -rc [real]         Cutoff radius" << endl;
    cout << "\t -ns [real]         Number of integration steps" << endl;
    cout << "\t -T0 [real]         Initial temperature" << endl;
    cout << "\t -thermof [integer] Thermo information print frequency" << endl;
    cout << "\t -outf [integer]    Print positions to XYZ file frequency" << endl;
    cout << "\t -onefile           Write config to single output file (multiple files otherwise)"
        << endl;
    cout << "\t -icf [string]      Initial configuration file" << endl;
    cout << "\t -ecorr             Use energy correction" << endl;
    cout << "\t -seed [integer]    Random number generator seed" << endl;
    cout << "\t -uf                Print unfolded coordinates in output files" << endl;
    cout << "\t -novelo            Not print and not read velocity from files" << endl;
    cout << "\t -h                 Print this info" << endl;
}

SystemOptions ParseCommandLineArguments(int argc, char** argv) {
    SystemOptions options;
    for (int arg_index = 1; arg_index < argc; ++arg_index) {
        string arg_str = argv[arg_index];
        if (arg_str == "-N") options.particles_number = atoi(argv[++arg_index]);
        else if (arg_str == "-rho") options.density = atof(argv[++arg_index]);
        else if (arg_str == "-dt") options.dt = atof(argv[++arg_index]);
        else if (arg_str == "-rc") options.cutoff_radius = atof(argv[++arg_index]);
        else if (arg_str == "-ns") options.steps_number = atoi(argv[++arg_index]);
        else if (arg_str == "-T0") options.T0 = atof(argv[++arg_index]);
        else if (arg_str == "-thermof") options.print_thermo_frequency = atoi(argv[++arg_index]);
        else if (arg_str == "-outf") options.print_out_frequency = atoi(argv[++arg_index]);
        else if (arg_str == "-onefile") options.write_output_in_one_file = true;
        else if (arg_str == "-icf") options.init_config_file = argv[++arg_index];
        else if (arg_str == "-ecorr") options.use_energy_correction = true;
        else if (arg_str == "-seed") options.seed = (unsigned)atoi(argv[++arg_index]);
        else if (arg_str == "-uf") options.print_unfolded_coordinates = true;
        else if (arg_str == "-novelo") options.read_and_print_with_velocity = false;
        else if (arg_str == "-h") {
            PrintUsageInfo();
            exit(0);
        }
        else {
            cerr << "Error: Command-line argument '" << arg_str << "' not recognized." << endl;
            exit(-1);
        }
    }

    return options;
}

vector<Particle> ReadParticlesXYZ(ifstream& in_file, SystemOptions& options) {
    in_file >> options.particles_number;

    // Here we need to parse the line like
    //    Lattice="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0"
    // And just get box_size from here
    string rubbish;
    in_file >> rubbish >> options.box_size;
    getline(in_file, rubbish);

    options.density = options.particles_number / 
        (options.box_size * options.box_size * options.box_size);

    vector<Particle> particles;
    for (int index = 0; index < options.particles_number; ++index) {
        Particle read_particle;
        if (options.read_and_print_with_velocity) {
            in_file >> read_particle.x >> read_particle.y >> read_particle.z
                >> read_particle.vx >> read_particle.vy >> read_particle.vz;
        }
        else {
            in_file >> read_particle.x >> read_particle.y >> read_particle.z;
        }

        // If something else is left in the line (for example velocity), do not read
        getline(in_file, rubbish);

        particles.push_back(read_particle);
    }
    
    return particles;
}

void WriteParticlesXYZ(ofstream& out_file, const vector<Particle>& particles, 
                       const SystemOptions& options, double time) {
    out_file << options.particles_number << endl;
    out_file << "Lattice=\" " << options.box_size << " 0.0 0.0 0.0 "
        << options.box_size << " 0.0 0.0 0.0 " << options.box_size << " \"";
    if (options.read_and_print_with_velocity) {
        out_file << " Properties=pos:R:3:velo:R:3";
    }
    else {
        out_file << " Properties=pos:R:3";
    }
    out_file << " Time = " << time << endl;
    for (const auto& particle : particles) {
        if (options.print_unfolded_coordinates) {
            out_file << particle.x + particle.image_x * options.box_size << " "
                << particle.y + particle.image_y * options.box_size << " "
                << particle.z + particle.image_z * options.box_size;
        }
        else {
            out_file << particle.x << " " << particle.y << " " << particle.z;
        }

        if (options.read_and_print_with_velocity) {
            out_file << " " << particle.vx << " " << particle.vy << " " << particle.vz;
        }

        out_file << endl;
    }
}

vector<Particle> GeneratePositions(const SystemOptions& options) {
    // Find the lowest perfect cube, n3, greater than or equal to the number of particles
    int lattice_size = 2;
    while (lattice_size * lattice_size * lattice_size < options.particles_number) {
        lattice_size++;
    }

    // Generate particles in simple cubic (sc) lattice
    vector<Particle> particles;
    int index_x = 0, index_y = 0, index_z = 0;
    for (int index = 0; index < options.particles_number; ++index) {
        Particle new_particle;
        new_particle.x = ((double)index_x + 0.5) * options.box_size / lattice_size;
        new_particle.y = ((double)index_y + 0.5) * options.box_size / lattice_size;
        new_particle.z = ((double)index_z + 0.5) * options.box_size / lattice_size;

        index_x++;
        if (index_x == lattice_size) {
            index_x = 0;
            index_y++;
            if (index_y == lattice_size) {
                index_y = 0;
                index_z++;
            }
        }

        particles.push_back(new_particle);
    }

    return particles;
}

void GenerateVelocities(vector<Particle>& particles, const SystemOptions& options, mt19937& rng) {
    normal_distribution<double> distribution(0, 1);
    
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

double ComputeForcesAndPotentialEnergy(vector<Particle>& particles, const SystemOptions& options) {
    for (auto& particle : particles) {
        particle.ax = particle.ay = particle.az = 0;
    }

    double half_box_size = options.box_size / 2;
    double squared_cutoff = options.cutoff_radius * options.cutoff_radius;
    double potential_energy = 0.0;

    for (int i = 0; i < options.particles_number - 1; ++i) {
        for (int j = i + 1; j < options.particles_number; ++j) {
            double delta_x = particles[i].x - particles[j].x;
            double delta_y = particles[i].y - particles[j].y;
            double delta_z = particles[i].z - particles[j].z;

            // Periodic boundary conditions: Apply the minimum image convention; note that
            // this is *not* used to truncate the potential as long as there an explicit cutoff.
            if (delta_x > half_box_size) {
                delta_x -= options.box_size;
            }
            else if (delta_x < -half_box_size) {
                delta_x += options.box_size;
            }
            if (delta_y > half_box_size) {
                delta_y -= options.box_size;
            }
            else if (delta_y < -half_box_size) {
                delta_y += options.box_size;
            }
            if (delta_z > half_box_size) {
                delta_z -= options.box_size;
            }
            else if (delta_z < -half_box_size) {
                delta_z += options.box_size;
            }

            double squared_distance = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

            if (squared_distance < squared_cutoff) {
                double distance_pow_6 = 1 / (squared_distance * squared_distance * squared_distance);
                potential_energy += 4 * (distance_pow_6 * distance_pow_6 - distance_pow_6) 
                    - options.energy_cut;
                double force = 48 * (distance_pow_6 * distance_pow_6 - 0.5 * distance_pow_6);

                particles[i].ax += delta_x * force / squared_distance;
                particles[j].ax -= delta_x * force / squared_distance;
                particles[i].ay += delta_y * force / squared_distance;
                particles[j].ay -= delta_y * force / squared_distance;
                particles[i].az += delta_z * force / squared_distance;
                particles[j].az -= delta_z * force / squared_distance;
            }
        }
    }

    return potential_energy + options.particles_number * options.energy_correction;
}

void ApplyPeriodicBoundaryConditions(vector<Particle>& particles, const SystemOptions& options) {
    for (auto& particle : particles) {
        if (particle.x < 0.0) {
            particle.x += options.box_size;
            particle.image_x--;
        }
        else if (particle.x > options.box_size) {
            particle.x -= options.box_size;
            particle.image_x++;
        }
        if (particle.y < 0.0) {
            particle.y += options.box_size;
            particle.image_y--;
        }
        else if (particle.y > options.box_size) {
            particle.y -= options.box_size;
            particle.image_y++;
        }
        if (particle.z < 0.0) {
            particle.z += options.box_size;
            particle.image_z--;
        }
        else if (particle.z > options.box_size) {
            particle.z -= options.box_size;
            particle.image_z++;
        }
    }
}

int main(int argc, char* argv[]) {
    SystemOptions options = ParseCommandLineArguments(argc, argv);
    options.box_size = cbrt(options.particles_number / options.density);
    double rr3 = 1 / (options.cutoff_radius * options.cutoff_radius * options.cutoff_radius);
    options.energy_cut = 4 * (rr3 * rr3 * rr3 * rr3 - rr3 * rr3);
    if (options.use_energy_correction) {
        options.energy_correction = 8 * M_PI * options.density * (rr3 * rr3 * rr3 / 9.0 - rr3 / 3.0);
    }

    mt19937 rng(options.seed);

    // Output some initial information
    cout << "# NVE MD Simulation of a Lennard - Jones fluid" << endl;
    cout << "# L = " << options.box_size << " rho = " << options.density << " N = "
        << options.particles_number << " r_cut = " << options.cutoff_radius << endl;
    cout << "# Steps number = " << options.steps_number << " seed = " << options.seed 
        << " dt = " << options.dt << endl;

    vector<Particle> particles;
    if (options.init_config_file != "") {
        cout << "# Reading file '" << options.init_config_file << "'" << endl;
        ifstream in_file;
        in_file.open(options.init_config_file);
        if (in_file.is_open()) {
            particles = ReadParticlesXYZ(in_file, options);
            in_file.close();
        }
        else {
            cout << "Error: Could not open file '" << options.init_config_file <<"'" << endl;
            return -1;
        }
    }
    else {
        particles = GeneratePositions(options);
        GenerateVelocities(particles, options, rng);
    }

    // Initial output to file
    ofstream out_file;
    if (options.write_output_in_one_file) {
        out_file.open("out.xyz");
        WriteParticlesXYZ(out_file, particles, options, 0.0);
    }
    else {
        out_file.open("0.xyz");
        WriteParticlesXYZ(out_file, particles, options, 0.0);
        out_file.close();
    }

    double potential_energy = ComputeForcesAndPotentialEnergy(particles, options);
    double kinetic_energy = 0.0;
    for (auto& particle : particles) {
        kinetic_energy += (particle.vx * particle.vx + particle.vy * particle.vy
            + particle.vz * particle.vz) / 2;
    }
    double total_energy_initial = potential_energy + kinetic_energy;

    cout << "# step time PE KE TE drift T" << endl;
    cout << 0 << " " << 0 * options.dt << " " << potential_energy << " "
        << kinetic_energy << " " << total_energy_initial << " "
        << 0 << " "
        << kinetic_energy * 2 / 3 / options.particles_number << endl;

    // The Great Integration Cycle
    for (unsigned step = 1; step <= options.steps_number; ++step) {
        // First integration half-step
        for (auto& particle : particles) {
            particle.x += particle.vx * options.dt + 0.5 * options.dt * options.dt * particle.ax;
            particle.y += particle.vy * options.dt + 0.5 * options.dt * options.dt * particle.ay;
            particle.z += particle.vz * options.dt + 0.5 * options.dt * options.dt * particle.az;

            particle.vx += 0.5 * options.dt * particle.ax;
            particle.vy += 0.5 * options.dt * particle.ay;
            particle.vz += 0.5 * options.dt * particle.az;
        }
        ApplyPeriodicBoundaryConditions(particles, options);

        potential_energy = ComputeForcesAndPotentialEnergy(particles, options);

        // Second integration half-step
        kinetic_energy = 0.0;
        for (auto& particle : particles) {
            particle.vx += 0.5 * options.dt * particle.ax;
            particle.vy += 0.5 * options.dt * particle.ay;
            particle.vz += 0.5 * options.dt * particle.az;

            kinetic_energy += (particle.vx * particle.vx + particle.vy * particle.vy 
                + particle.vz * particle.vz) / 2;
        }

        double total_energy = potential_energy + kinetic_energy;

        if (step % options.print_thermo_frequency == 0) {
            cout << step << " " << step * options.dt << " " << potential_energy << " "
                << kinetic_energy << " " << total_energy << " "
                << (total_energy - total_energy_initial) / total_energy_initial << " "
                << kinetic_energy * 2 / 3 / options.particles_number << endl;
        }
        if (step % options.print_out_frequency == 0) {
            if (options.write_output_in_one_file) {
                WriteParticlesXYZ(out_file, particles, options, 0.0);
            }
            else {
                out_file.open(to_string(step) + ".xyz");
                WriteParticlesXYZ(out_file, particles, options, 0.0);
                out_file.close();
            }
        }
    }

    if (options.write_output_in_one_file) {
        out_file.close();
    }
}
