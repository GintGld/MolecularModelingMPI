#include "mdlj.h"
#include "mdlj-mpi.h"

using std::cout;
using std::ofstream;
using std::vector;

int main(int argc, char** argv) {
    // initialize system options and mpi
    ParseCommandLineArguments(argc, argv);
    init_cart_comm(argc, argv);

    GeneratePositions_MPI();
    
    double start = MPI_Wtime();

    Simulate_MPI();

    double stop = MPI_Wtime();

    if (MPI_OPTIONS.rank == 0) {
        cout << "particles:\t" << OPTIONS.global_particles_number << "\n"
            << "Cells:\t\t" << MPI_OPTIONS.cells << "\n" 
            << "particles\nper cell:\t" << (double)(OPTIONS.global_particles_number) / (double)(MPI_OPTIONS.cells) << "\n"
            << "Steps:\t\t" << OPTIONS.steps_number << "\n"
            << "Time:\t\t" << stop - start << "\n"
            << "Precision:\t" << MPI_Wtick() << "\n";
    }

    // finilize mpi
    finilize_cart_comm();
}