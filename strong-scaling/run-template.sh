#!/bin/bash

#SBATCH --ntasks=$${tasks}                     # Количество MPI процессов
#SBATCH --tasks-per-node=$${node}
#SBATCH --comment "OpenMPI project"
#SBATCH -p RT_study

srun ../mpi/main -rho 0.01 -T0 0.1 -N 512 -ns 100000 -dt 0.001 -dx $${x} -dy $${y} -dz $${z} >> $${file}