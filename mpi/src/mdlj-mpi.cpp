#include "mdlj-mpi.h"
#define ROOT_PROCESS 0

using std::cout;
using std::string;
using std::to_string;
using std::vector;
using std::ofstream;

MPI_Comm COMM;
__mpi_options MPI_OPTIONS;

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
    MPI_Type_contiguous(12, MPI_DOUBLE, &MPI_OPTIONS.dt_particles);
    MPI_Type_commit(&MPI_OPTIONS.dt_particles);

    // Create Cartesian Commutator
    int reorder = 0;
    int dims[3] = {OPTIONS.dimx, OPTIONS.dimy, OPTIONS.dimz}, periods[3] = {1,1,1};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &COMM);
    
    // Get rank of the process
    MPI_Cart_map(COMM, 3, dims, periods, &MPI_OPTIONS.rank);

    // Get process coordinates
    MPI_Cart_coords(COMM, MPI_OPTIONS.rank, 3, MPI_OPTIONS.coords);
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
    // (coordinates are relative)
    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        // create vector of vectors to separate
        // particles for corresponding cells
        vector< vector<Particle> > particles_divided_by_cells = __generate_positions_per_cell();

        __generate_velocities(particles_divided_by_cells);

        // Construct info arrays for `MPI_Scatterv` from vector
        counts = new int[MPI_OPTIONS.cells];
        displs = new int[MPI_OPTIONS.cells];
        for (int i = 0; i < MPI_OPTIONS.cells; ++i) {
            counts[i] = particles_divided_by_cells[i].size();
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        }

        // flatten particles
        sendbuff = new Particle[OPTIONS.global_particles_number];
        for (int i = 0; i < MPI_OPTIONS.cells; ++i) {
            for (int j = 0; j < counts[i]; ++j) {
                sendbuff[displs[i] + j] = particles_divided_by_cells[i][j];
            }
        }
    }

    // send number of particles for each cell
    MPI_Scatter(counts, 1, MPI_INT,
                &number_of_particles, 1, MPI_INT, 
                ROOT_PROCESS, COMM);

    // allocate memory for input
    particles_for_current_process = new Particle[number_of_particles];

    // send particles for cells
    MPI_Scatterv(sendbuff, counts, displs, MPI_OPTIONS.dt_particles,
                 particles_for_current_process, number_of_particles, MPI_OPTIONS.dt_particles,
                 ROOT_PROCESS, COMM);

    // push collected data to vector
    particles.clear();
    for (int i = 0; i < number_of_particles; ++i)
        particles.push_back(particles_for_current_process[i]);

    delete[] particles_for_current_process;
    delete[] sendbuff;
    delete[] counts;
    delete[] displs;
}

void WriteParticlesXYZ_MPI(ofstream& stream, double time) {
    /*
     * Collect particles from all cells and
     * Write them using `__write_particlesXYZ`
    */
    // Pointers for `MPI_Gatherv`, neccecary only in root process
    Particle *rbuff = nullptr;
    int *counts = nullptr, *displs = nullptr;

    // allocate memory for `counts` array
    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        counts = new int[MPI_OPTIONS.cells];
    }

    // get number of particles in each cell
    int particles_per_cell = particles.size();
    MPI_Gather(&particles_per_cell, 1, MPI_INT,
               counts, 1, MPI_INT,
               ROOT_PROCESS, COMM);

    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        // allocate memory for `displs` array
        displs = new int[MPI_OPTIONS.cells];
        for(int i = 0; i < MPI_OPTIONS.cells; ++i)
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        
        // allocate memory for `recvbuf` array
        rbuff = new Particle[OPTIONS.global_particles_number];
    }

    // Get particles from all cells
    MPI_Gatherv(particles.data(), particles.size(), MPI_OPTIONS.dt_particles,
                rbuff, counts, displs, MPI_OPTIONS.dt_particles,
                ROOT_PROCESS, COMM);

    // convert relative coordinates to absolute ones
    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        int coords[3];
        for (int rank = 0; rank < MPI_OPTIONS.cells; ++rank) {
            MPI_Cart_coords(COMM, rank, 3, coords);
            for (int i = displs[rank]; i < displs[rank] + counts[rank]; ++i) {
                rbuff[i].x += coords[0] * OPTIONS.simple_box_size;
                rbuff[i].y += coords[1] * OPTIONS.simple_box_size;
                rbuff[i].z += coords[2] * OPTIONS.simple_box_size;
            }
        }
    }

    if (MPI_OPTIONS.rank == ROOT_PROCESS) {
        __write_particles_XYZ(stream, rbuff, time);
    }

    delete[] counts;
    delete[] displs;
    delete[] rbuff;
}

vector< vector<Particle> >
__get_neighbor_particles_MPI() {
    /*
     * Get neighbor particles to calculate forces
     * returned coordinates are relative
    */
    int displs[6], counts[6];

    // get number of particles from each neighbor (6 neighbors)
    int number_of_particles = particles.size(), neighbor_number = 0;
    MPI_Neighbor_allgather(&number_of_particles, 1, MPI_INT,
                           counts, 1, MPI_INT,
                           COMM);

    // make info arrays
    for (int i = 0; i < 6; ++i) {
        neighbor_number += counts[i];
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    // get particles from particles
    Particle* buff = new Particle[neighbor_number];
    MPI_Neighbor_allgatherv(particles.data(), particles.size(), MPI_OPTIONS.dt_particles,
                            buff, counts, displs, MPI_OPTIONS.dt_particles,
                            COMM);

    // convert to vector
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
    vector< vector<int> > particles_to_send(6);
    vector<int> particles_to_delete;

    // collect particles id to send
    // and fix their coordinates
    // to be correct in new cell
    for (unsigned i = 0; i < particles.size(); ++i) {
        if (particles[i].x < 0) {
            particles[i].x += OPTIONS.simple_box_size;
            particles[i].image_x -= 1;
            particles_to_send[0].push_back(i);
            particles_to_delete.push_back(i);
            continue;
        }
        if (particles[i].x > OPTIONS.simple_box_size) {
            particles[i].x -= OPTIONS.simple_box_size;
            particles[i].image_x += 1;
            particles_to_send[1].push_back(i);
            particles_to_delete.push_back(i);
            continue;
        }
        if (particles[i].y < 0) {
            particles[i].y += OPTIONS.simple_box_size;
            particles[i].image_y -= 1;
            particles_to_send[2].push_back(i);
            particles_to_delete.push_back(i);
            continue;
        }
        if (particles[i].y > OPTIONS.simple_box_size) {
            particles[i].y -= OPTIONS.simple_box_size;
            particles[i].image_y += 1;
            particles_to_send[3].push_back(i);
            particles_to_delete.push_back(i);
            continue;
        }
        if (particles[i].z < 0) {
            particles[i].z += OPTIONS.simple_box_size;
            particles[i].image_z -= 1;
            particles_to_send[4].push_back(i);
            particles_to_delete.push_back(i);
            continue;
        }
        if (particles[i].z > OPTIONS.simple_box_size) {
            particles[i].z -= OPTIONS.simple_box_size;
            particles[i].image_z += 1;
            particles_to_send[5].push_back(i);
            particles_to_delete.push_back(i);
            continue;
        }
    }

    // number of particles to send
    int send_size = particles_to_send[0].size() + 
                    particles_to_send[1].size() + 
                    particles_to_send[2].size() + 
                    particles_to_send[3].size() + 
                    particles_to_send[4].size() + 
                    particles_to_send[5].size();

    // generate send info
    Particle* sbuff = (send_size == 0) ? nullptr : new Particle[send_size];
    int scounts[6], sdispls[6];
    for (int i = 0; i < 6; ++i) {
        scounts[i] = particles_to_send[i].size();
        sdispls[i] = (i == 0) ? 0 : sdispls[i - 1] + scounts[i - 1];

        for (int j = 0; j < scounts[i]; ++j)
            sbuff[sdispls[i] + j] = particles[particles_to_send[i][j]];
    }

    // get number of particles to load
    int rcounts[6], rdispls[6];
    MPI_Neighbor_alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, COMM);

    // generate receive displs
    for (int i = 0; i < 6; ++i)
        rdispls[i] = (i == 0) ? 0 : rdispls[i - 1] + rcounts[i - 1];

    // collect neighbors particles
    int number_to_load = rdispls[5] + rcounts[5];
    Particle* rbuff = (number_to_load == 0) ? nullptr : new Particle[number_to_load];
    MPI_Neighbor_alltoallv(sbuff, scounts, sdispls, MPI_OPTIONS.dt_particles,
                           rbuff, rcounts, rdispls, MPI_OPTIONS.dt_particles, COMM);

    // delete sended particles
    for(int i = particles_to_delete.size() - 1; i >= 0; --i)
        particles.erase(particles.begin() + particles_to_delete[i]);

    // insert received particles
    if (number_to_load > 0)
        particles.insert(particles.end(), rbuff, rbuff + number_to_load);

    delete[] sbuff;
    delete[] rbuff;
}

void __compute_forces_MPI() {
    for (auto& p : particles)
        p.ax = p.ay = p.az = 0;

    vector< vector<Particle> > neighbors = __get_neighbor_particles_MPI();

    // double half_box_size = OPTIONS.simple_box_size / 2;
    double squared_cutoff = OPTIONS.cutoff_radius * OPTIONS.cutoff_radius;
    potential_energy = 0.0;

    for (unsigned i = 0; i < particles.size(); ++i) {
        // iterate over process particles
        for (unsigned j = i + 1; j < particles.size(); ++j) {
            double delta_x = particles[i].x - particles[j].x;
            double delta_y = particles[i].y - particles[j].y;
            double delta_z = particles[i].z - particles[j].z;

            double squared_distance = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

            if (squared_distance < squared_cutoff) {
                double distance_pow_6 = 1 / (squared_distance * squared_distance * squared_distance);
                potential_energy += 4 * (distance_pow_6 * distance_pow_6 - distance_pow_6) 
                    - OPTIONS.energy_cut;
                double force = 48 * (distance_pow_6 * distance_pow_6 - 0.5 * distance_pow_6);

                particles[i].ax += delta_x * force / squared_distance;
                particles[j].ax -= delta_x * force / squared_distance;
                particles[i].ay += delta_y * force / squared_distance;
                particles[j].ay -= delta_y * force / squared_distance;
                particles[i].az += delta_z * force / squared_distance;
                particles[j].az -= delta_z * force / squared_distance;
            }
        }
        // iterate over neighbor particles
        for (int shift = 0; shift < 6; ++shift) {
            for (unsigned j = 0; j < neighbors[shift].size(); ++j) {
                double delta_x = particles[i].x - neighbors[shift][j].x;
                double delta_y = particles[i].y - neighbors[shift][j].y;
                double delta_z = particles[i].z - neighbors[shift][j].z;

                // apply shift between cells
                switch (shift) {
                    case 0:
                        delta_x += OPTIONS.simple_box_size;
                        break;
                    case 1:
                        delta_x -= OPTIONS.simple_box_size;
                        break;
                    case 2:
                        delta_y += OPTIONS.simple_box_size;
                        break;
                    case 3:
                        delta_y -= OPTIONS.simple_box_size;
                        break;
                    case 4:
                        delta_z += OPTIONS.simple_box_size;
                        break;
                    case 5:
                        delta_z -= OPTIONS.simple_box_size;
                        break;
                }

                double squared_distance = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

                if (squared_distance < squared_cutoff) {
                    double distance_pow_6 = 1 / (squared_distance * squared_distance * squared_distance);
                    potential_energy += 4 * (distance_pow_6 * distance_pow_6 - distance_pow_6) 
                        - OPTIONS.energy_cut;
                    double force = 48 * (distance_pow_6 * distance_pow_6 - 0.5 * distance_pow_6);

                    particles[i].ax += delta_x * force / squared_distance;
                    particles[i].ay += delta_y * force / squared_distance;
                    particles[i].az += delta_z * force / squared_distance;
                }
            }
        }
    }
}

void __make_simulation_step_MPI() {
    __first_half_step();
    __apply_periodic_boundary_conditions_MPI();
    __compute_forces_MPI();
    __second_half_step(); 
}

void Simulate_MPI() {
    // Initial output to file
    ofstream out_file;
    out_file.open("tmp/out.xyz");
    WriteParticlesXYZ_MPI(out_file, 0);
    
    __compute_forces_MPI();
    
    // kinetic_energy = 0.0;
    // for (auto& p : particles)
    //     kinetic_energy += (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz) / 2;
    // double total_energy_initial = potential_energy + kinetic_energy;

    // cout << "# step time PE KE TE drift T" << "\n";
    // cout << 0 << " " << 0 << " " << potential_energy << " "
    //     << kinetic_energy << " " << total_energy << " " << 0 << " "
    //     << kinetic_energy * 2 / 3 / OPTIONS.global_particles_number << "\n";

    for (unsigned step = 1; step <= OPTIONS.steps_number; ++step) {
        __make_simulation_step_MPI();

        // total_energy = potential_energy + kinetic_energy;

        // if (step % OPTIONS.print_thermo_frequency == 0) {
        //     cout << step << " " << step * OPTIONS.dt << " " << potential_energy << " "
        //         << kinetic_energy << " " << total_energy << " "
        //         << (total_energy - total_energy_initial) / total_energy_initial << " "
        //         << kinetic_energy * 2 / 3 / OPTIONS.global_particles_number << "\n";
        // }
        if (step % OPTIONS.print_out_frequency == 0)
            WriteParticlesXYZ_MPI(out_file, 0.0);
    }

    out_file.close();
}