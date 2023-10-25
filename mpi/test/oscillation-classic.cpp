#include "mdlj.h"
#include "mdlj-mpi.h"

using std::cout;
using std::ofstream;
using std::vector;

int main(int argc, char** argv) {
    // initialize system options and mpi
    ParseCommandLineArguments(argc, argv);
    init_cart_comm(argc, argv);

    if (OPTIONS.dimx != 1 || OPTIONS.dimy != 1 || OPTIONS.dimz != 1) {
        cout << "Wrong cells for this test. Expected [2, 1, 1].\n";
        return 1;
    }

    double custom_cell_size = 10;
    
    OPTIONS.global_particles_number = 2;
    OPTIONS.print_out_frequency = 1;
    OPTIONS.simple_box_size = custom_cell_size;
    OPTIONS.steps_number = 1000;
    OPTIONS.dt = 0.01;
    
    particles = vector<Particle>{Particle{custom_cell_size / 2 - 0.6, custom_cell_size / 2, custom_cell_size / 2},
            {custom_cell_size / 2 + 0.6, custom_cell_size / 2, custom_cell_size / 2}};
    
    Simulate_MPI();

    // finilize mpi
    finilize_cart_comm();
}