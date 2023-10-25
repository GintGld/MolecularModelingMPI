#include "mdlj.h"
#include "mdlj-mpi.h"

using std::cout;
using std::ofstream;
using std::vector;

int main(int argc, char** argv) {
    // initialize system options and mpi
    ParseCommandLineArguments(argc, argv);
    init_cart_comm(argc, argv);

    if (OPTIONS.dimx != 2 || OPTIONS.dimy != 2 || OPTIONS.dimz != 2) {
        cout << "Wrong cells for this test. Expected [2, 2, 2].\n";
        return 1;
    }

    double custom_cell_size = 0.1;
    
    OPTIONS.global_particles_number = 1;
    OPTIONS.print_out_frequency = 1;
    OPTIONS.simple_box_size = custom_cell_size;
    OPTIONS.steps_number = 100;
    OPTIONS.dt = 0.01;
    
    if (MPI_OPTIONS.coords[0] == 1 && MPI_OPTIONS.coords[1] == 1 && MPI_OPTIONS.coords[2] == 1) {
        Particle p{custom_cell_size / 2, custom_cell_size / 2, custom_cell_size * 3 / 4};
        p.vx = 1; p.vy = 1; p.vz = 1;
        particles = {p};
    }
    
    Simulate_MPI();

    // finilize mpi
    finilize_cart_comm();
}