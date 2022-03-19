#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define array_size 1610612736
#define num_threads 512

double* input_data = NULL;
double* output_data = NULL;
double my_sum;

extern void initCuda(int my_rank, int num_elements);
extern void reduceCuda(int size, int threads, int blocks, double *d_idata, double *d_odata);

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

    num_elements = 10;

    printf("Rank %d has %d sized block\n", my_rank, num_elements);

    initCuda(my_rank, num_elements);

    reduceCuda(num_elements, num_threads, num_elements/num_threads, input_data, output_data);
    printf("Rank %d has %f\n", my_rank, my_sum);

    double result = 0;
    MPI_Reduce(&my_sum, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(my_rank == 0){
        printf("Result %f\n", result);
    }

    MPI_Finalize();
}
