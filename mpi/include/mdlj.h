/*
 * Header for single-process functions
*/
#pragma once
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>

struct Particle {
    double x, y, z;
    double vx = 0, vy = 0, vz = 0;
    double ax = 0, ay = 0, az = 0;
    double image_x = 0, image_y = 0, image_z = 0;
};

struct __system_options {
    int dimx, dimy, dimz;
    unsigned global_particles_number = 216;
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
    bool use_energy_correction = false;
    bool read_and_print_with_velocity = true;
};

extern __system_options OPTIONS;
extern std::vector<Particle> particles;
extern double potential_energy;
extern double kinetic_energy;
extern double total_energy;

void PrintUsageInfo();
void ParseCommandLineArguments(int argc, char** argv);

// non public function, used only from `GeneratePositions_MPI`
std::vector< std::vector<Particle> > __generate_positions_per_cell();

void GenerateVelocities(std::mt19937& rng);

// non public function, used only from `WriteParticlesXYZ_MPI`
void __write_particles_XYZ(std::ofstream& stream, Particle* buff,double time);

void __acceleration_zero();
void __first_half_step();
void __second_half_step();