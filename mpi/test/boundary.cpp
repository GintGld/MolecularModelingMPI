#include "mdlj.h"
#include "mdlj-mpi.h"

using std::cout;
using std::ofstream;
using std::vector;

int main(int argc, char** argv) {
    // initialize system options and mpi
    ParseCommandLineArguments(argc, argv);
    init_cart_comm(argc, argv);

    if (MPI_OPTIONS.cells != 2) {
        cout << "Invalid number of processes. Expected 2.\n";
        return 1;
    }

    int coords[3];
    OPTIONS.global_particles_number = 1;
    if (MPI_OPTIONS.rank == 0) {
        MPI_Cart_coords(COMM, 1, 3, coords);
        particles = {
            Particle{
                ((double)(coords[0]) + 0.5) * OPTIONS.simple_box_size,
                ((double)(coords[1]) + 0.5) * OPTIONS.simple_box_size,
                ((double)(coords[2]) + 0.5) * OPTIONS.simple_box_size
            }
        };
    }

    if (MPI_OPTIONS.rank == 0) {
        cout << "send particle " << particles[0].x << ' ' << particles[0].y << ' ' << particles[0].z << " "
             << "from proc_0\n";
    }

    __apply_periodic_boundary_conditions_MPI();

    if (MPI_OPTIONS.rank == 1) {
        cout << "get particle " << particles[0].x << ' ' << particles[0].y << ' ' << particles[0].z << ' '
             << "at proc_1\n";
    }

    // finilize mpi
    finilize_cart_comm();
}