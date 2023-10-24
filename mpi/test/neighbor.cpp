#include "mdlj.h"
#include "mdlj-mpi.h"

using std::cout;
using std::ofstream;
using std::vector;

int main(int argc, char** argv) {
    // initialize system options and mpi
    ParseCommandLineArguments(argc, argv);
    init_cart_comm(argc, argv);

    // Create particles
    GeneratePositions_MPI();

    auto v = __get_neighbor_particles_MPI();

    cout << MPI_OPTIONS.rank << "\n";
    for (int i = 0; i < v.size(); ++i) {
        for (const auto& p : v[i])
            cout << p.x << ' ' << p.y << ' ' << p.z << "\n";
        cout << "\n";
    }

    // finilize mpi
    finilize_cart_comm();
}