#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


void launch_kernels(int my_rank, int world_size);

int main(int argc, char** argv){

    int my_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    launch_kernels(my_rank, world_size);

    MPI_Finalize();
}
