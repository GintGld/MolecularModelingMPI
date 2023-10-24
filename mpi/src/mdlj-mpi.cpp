#include "mdlj-mpi.h"
#define ROOT_PROCESS 0

using std::cout;
using std::string;
using std::to_string;
using std::vector;
using std::ofstream;

MPI_Comm COMM;
__mpi_options MPI_OPTIONS;

void MPI_Apply(int ierr, string error_message) {
    /*
     * Wrapper for all MPI functions that
     * return error status as output
     * (almost all according to documentation)
    */
    if (ierr != MPI_SUCCESS) {
        cout << "USER CATCHED ERROR [" << ierr << "]\n"
             << error_message << "\n";
        MPI_Abort(COMM, EXIT_FAILURE);
    }
}

void init_cart_comm(int argc, char** argv) {
    /*
     * create Cartesian communicator and
     * initiate `MPIOptions` member that will save
     * all information about current process configuration
    */

    MPI_Init(&argc, &argv);

    // check if number of workers equals number of cells
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_OPTIONS.cells);
    int cells_theory = OPTIONS.dimx * OPTIONS.dimy * OPTIONS.dimz;
    if (MPI_OPTIONS.cells != cells_theory) {
        cout << "ERROR: number of processes not equal to number of cells.\n"
            << "Processes:\t" << MPI_OPTIONS.cells << "\n"
            << "Cells:\t\t" << cells_theory << "\n";
        MPI_Abort(COMM, EXIT_FAILURE);
    }

    // init particle data type
    MPI_Apply(
        MPI_Type_contiguous(12, MPI_DOUBLE, &MPI_OPTIONS.dt_particles),
        "Fail in creating particle mpi-data type"
    );
    MPI_Apply(
        MPI_Type_commit(&MPI_OPTIONS.dt_particles),
        "Fail in commiting particle mpi-data type"
    );

    // Create Cartesian Commutator
    int reorder = 0;
    int dims[3] = {OPTIONS.dimx, OPTIONS.dimy, OPTIONS.dimz}, periods[3] = {1,1,1};
    MPI_Apply(
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &COMM),
        "Fail to create Cartesian communicator"
    );
    
    // Get rank of the process
    MPI_Apply(
        MPI_Cart_map(COMM, 3, dims, periods, &MPI_OPTIONS.rank),
        string("Fail in getting rank of the process.\n")
    );

    // Get process coordinates
    MPI_Apply(
        MPI_Cart_coords(COMM, MPI_OPTIONS.rank, 3, MPI_OPTIONS.coords),
        string("fail in getting coordinates.\n") +
        string("rank:\t") + to_string(MPI_OPTIONS.rank)
    );
}

void finilize_cart_comm() {
    // Free the datatype
    MPI_Type_free(&MPI_OPTIONS.dt_particles);
 
    // Finish with mpi
    MPI_Finalize();
}

void GeneratePositions_MPI() {
    /*
     * Generates all particles in root process memory
     * then send it to all other processes
    */
    
    int number_of_particles;
    
    // pointers for `MPI_Scatterv`
    Particle *sendbuff = nullptr, *particles_for_current_process = nullptr;
    int *counts = nullptr, *displs = nullptr;

    // root process generates all particles in a sc1 and 
    // scatters particles for their cells
    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        // create vector of vectors to separate
        // particles for corresponding cells
        vector< vector<Particle> > particles_divided_by_cells = __generate_positions_per_cell();

        // Construct info arrays for `MPI_Scatterv` from vector
        counts = new int[MPI_OPTIONS.cells];
        displs = new int[MPI_OPTIONS.cells];
        for (int i = 0; i < MPI_OPTIONS.cells; ++i) {
            counts[i] = particles_divided_by_cells[i].size();
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        }

        // flatten particles
        sendbuff = new Particle[OPTIONS.particles_number];
        for (int i = 0; i < MPI_OPTIONS.cells; ++i) {
            for (int j = 0; j < counts[i]; ++j) {
                sendbuff[displs[i] + j] = particles_divided_by_cells[i][j];
           
            }
        }
    }

    // send number of particles for each cell
    MPI_Apply(
       MPI_Scatter(counts, 1, MPI_INT,
                   &number_of_particles, 1, MPI_INT, 
                   ROOT_PROCESS, COMM),
       string("Fail in GeneratePositions_MPI -> MPI_Scatter\n") + 
       "process rank\t" + to_string(MPI_OPTIONS.rank)
    );
    
    // allocate memory for input
    particles_for_current_process = new Particle[number_of_particles];

    // send particles for cells
    MPI_Apply(
        MPI_Scatterv(sendbuff, counts, displs, MPI_OPTIONS.dt_particles,
                     particles_for_current_process, number_of_particles, MPI_OPTIONS.dt_particles,
                     ROOT_PROCESS, COMM),
        string("Fail in GeneratePositions_MPI -> MPI_Scatter\n") + 
        "process rank\t" + to_string(MPI_OPTIONS.rank)
    );

    // push collected data to vector
    particles.clear();
    for (int i = 0; i < number_of_particles; ++i)
        particles.push_back(particles_for_current_process[i]);

    delete[] particles_for_current_process;
    delete[] sendbuff;
    delete[] counts;
    delete[] displs;
}

void WriteParticlesXYZ_MPI(ofstream& stream) {
    /*
     * Collect particles from all cells and
     * Write them using `WriteParticlesXYZ`
    */
    // Pointers for `MPI_Gatherv`, neccecary only in root process
    Particle *GatheredParticles = nullptr;
    int *counts = nullptr, *displs = nullptr;

    // allocate memory for `counts` array
    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        counts = new int[MPI_OPTIONS.cells];
    }

    // get number of particles in each cell
    int particles_per_cell = particles.size();
    MPI_Apply(
        MPI_Gather(&particles_per_cell, 1, MPI_INT,
                   counts, 1, MPI_INT,
                   ROOT_PROCESS, COMM),
        string("Fail in WriteParticlesXYZ_MPI -> MPI_Gather\n") + 
        "process rank\t" + to_string(MPI_OPTIONS.rank)
    );

    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        // allocate memory for `displs` array
        displs = new int[MPI_OPTIONS.cells];
        for(int i = 0; i < MPI_OPTIONS.cells; ++i)
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        
        // allocate memory for `recvbuf` array
        GatheredParticles = new Particle[OPTIONS.particles_number];
    }

    // Get particles from all cells
    MPI_Apply(
        MPI_Gatherv(particles.data(), particles.size(), MPI_OPTIONS.dt_particles,
                    GatheredParticles, counts, displs, MPI_OPTIONS.dt_particles,
                    ROOT_PROCESS, COMM),
        string("Fail in WriteParticlesXYZ_MPI -> MPI_Gatherv\n") + 
        "process rank\t" + to_string(MPI_OPTIONS.rank)
    );

    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        __write_particles_XYZ(stream, GatheredParticles);
    }

    delete[] counts;
    delete[] displs;
    delete[] GatheredParticles;
}

vector< vector<Particle> >
__get_neighbor_particles_MPI() {
    /*
     * Get neighbor particles to calculate forces
    */
    int displs[6], counts[6];

    // get number of particles from each neighbor (6 neighbors)
    int number_of_particles = particles.size(), neighbor_number = 0;
    MPI_Apply(
        MPI_Neighbor_allgather(&number_of_particles, 1, MPI_INT,
                               counts, 1, MPI_INT,
                               COMM),
        string("Fail in __get_neighbor_particles_MPI -> MPI_Neighbor_allgather.\n") + 
        "process rank\t" + to_string(MPI_OPTIONS.rank)
    );

    for (int i = 0; i < 6; ++i) {
        neighbor_number += counts[i];
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    // cout << MPI_OPTIONS.rank << ' ' << neighbor_number << "\n";
    // for (int i = 0; i < 6; ++i)
    //     cout << counts[i] << ' ' << displs[i] << "\n";

    Particle* buff = new Particle[neighbor_number];

    MPI_Apply(
        MPI_Neighbor_allgatherv(particles.data(), particles.size(), MPI_OPTIONS.dt_particles,
                                buff, counts, displs, MPI_OPTIONS.dt_particles,
                                COMM),
        string("Fail in __get_neighbor_particles_MPI -> MPI_Neighbor_allgather.\n") + 
        "process rank\t" + to_string(MPI_OPTIONS.rank)
    );

    vector< vector<Particle> > neighbor_particles(6);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < counts[i]; ++j) {
            neighbor_particles[i].push_back( buff[displs[i] + j] );
        }
    }
    
    delete[] buff;

    return neighbor_particles;
}

void __apply_periodic_boundary_conditions_MPI() {

}

void __compute_forces_MPI() {

    __acceleration_zero();
}

void MakeSimulationStep_MPi() {

}