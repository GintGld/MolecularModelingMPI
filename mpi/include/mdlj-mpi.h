/*
 * Header for multi-process functions
 * All functions here are wrappers
 * for MPI library or "mdlj.h" functions
*/
#pragma once
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <mpi.h>

#include "mdlj.h"

struct __mpi_options {
    MPI_Datatype dt_particles;
    int cells, rank, coords[3];
};

extern MPI_Comm COMM;
extern __mpi_options MPI_OPTIONS;

void init_cart_comm(int argc, char** argv);
void finilize_cart_comm();

void GeneratePositions_MPI();
void WriteParticlesXYZ_MPI(std::ofstream& stream, double time = 0);

std::vector< std::vector<Particle> >
__get_neighbor_particles_MPI();
void __apply_periodic_boundary_conditions_MPI();
void __compute_forces_MPI();
void __make_simulation_step_MPI();
void Simulate_MPI();