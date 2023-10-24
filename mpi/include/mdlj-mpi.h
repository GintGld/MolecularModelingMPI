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
    MPI_Comm COMM = MPI_COMM_WORLD;
    int cells, rank, coords[3], neighbors[3][2];
};

extern MPI_Comm COMM;
extern __mpi_options MPI_OPTIONS;

void MPI_Apply(int ierr, std::string error_message);
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