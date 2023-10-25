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
    
    Simulate_MPI();

    // finilize mpi
    finilize_cart_comm();
}