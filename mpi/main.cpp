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
    std::mt19937 rng;
    GenerateVelocities(rng);

    auto v = __get_neighbor_particles_MPI();

    cout << MPI_OPTIONS.rank << "\n";
    for (int i = 0; i < v.size(); ++i) {
        for (const auto& p : v[i])
            cout << p.x << ' ' << p.y << ' ' << p.z << "\n";
        cout << "\n";
    }

    // // Write particles
    // ofstream out;
    // WriteParticlesXYZ_MPI(out);

    // finilize mpi
    finilize_cart_comm();
}