#include "mdlj.h"
#include "mdlj-mpi.h"

using std::cout;
using std::ofstream;
using std::vector;

int main(int argc, char** argv) {
    // initialize system options and mpi
    ParseCommandLineArguments(argc, argv);
    init_cart_comm(argc, argv);

    if (OPTIONS.dimx != 2 || OPTIONS.dimy != 2 || OPTIONS.dimz != 1) {
        cout << "Wrong cells for this test. Expected [2, 2, 1].\n";
        return 1;
    }

    double custom_cell_size = 10;
    double delta = 0.1;
    
    OPTIONS.global_particles_number = 2;
    OPTIONS.print_out_frequency = 1;
    OPTIONS.simple_box_size = custom_cell_size;
    OPTIONS.steps_number = 5;
    OPTIONS.dt = 0.001;
    
    if (MPI_OPTIONS.coords[0] == 0 && MPI_OPTIONS.coords[1] == 0 && MPI_OPTIONS.coords[2] == 0) {
        Particle p{custom_cell_size - delta, custom_cell_size, custom_cell_size / 2};
        p.vy = 0.1;
        particles = {p};
    } else if (MPI_OPTIONS.coords[0] == 1 && MPI_OPTIONS.coords[1] == 1 && MPI_OPTIONS.coords[2] == 0) {
        particles = {Particle{delta, delta, custom_cell_size / 2}};
    }

    Simulate_MPI();

    // finilize mpi
    finilize_cart_comm();
}