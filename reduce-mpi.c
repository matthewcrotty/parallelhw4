#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char** argv){

    int my_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if( (cE = cudaGetDiviceCound(&cudaDeviceCount)) != cudaSuccess){
        printf(" Unable to determine cuda device coucnt, error is %d, count is %d\n", cE, cudaDeviceCount);
        exit(-1);
    }

    if( (cE = cudaSetDevice(my_rank % cudaDeviceCount)) != cudaSucess){
        printF(" Unable to have rank %d set to cuda device %d, error is %d \n", my_rank, (my_rank % cudaDeviceCount), cE);
        exit(-1);
    }

    MPI_Finalize();
}
