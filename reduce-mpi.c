#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define array_size 1610612736

double launch_kernels(int my_rank, int world_size);

int main(int argc, char** argv){

    int my_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_elements = 0;
    if(my_rank == world_size-1){
        num_elements = array_size/world_size + array_size % (world_size);
    } else {
        num_elements = array_size/world_size;
    }

    printf("Rank %d has %d sized block\n", my_rank, num_elements);

    double result = launch_kernels(my_rank, world_size);

    if(my_rank == 0){
        printf("Result %f\n", result);
    }

    MPI_Finalize();
}
