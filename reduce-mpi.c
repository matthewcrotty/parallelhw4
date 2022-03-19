#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "clockcycle.h"

#define array_size 1610612736
#define num_threads 1024

double* input_data = NULL;
double* output_data = NULL;
double my_sum;

extern void initCuda(int my_rank, int num_elements);
extern void reduceCuda(int size, int threads, int blocks, double *d_idata, double *d_odata);

int main(int argc, char** argv){

    int my_rank;
    int world_size;

    unsigned long long start_time;
    unsigned long long end_time;
    int clock_frequency = 512000000;
    double time_in_secs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Calculate how many elelments per rank
    int num_elements = 0;
    if(my_rank == world_size-1){
        num_elements = array_size/world_size + array_size % (world_size);
    } else {
        num_elements = array_size/world_size;
    }

    initCuda(my_rank, num_elements);

    // Timed Section of computation
    start_time = clock_now();
    reduceCuda(num_elements, num_threads, num_elements/num_threads, input_data, output_data);
    double result = 0;
    MPI_Reduce(&my_sum, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    end_time = clock_now();

    MPI_Barrier(MPI_COMM_WORLD);

    if(my_rank == 0){
        time_in_secs = ((double)(end_time - start_time)) / clock_frequency;
        printf("Result %f\n", result);
        printf("Time %f\n", time_in_secs);
    }

    MPI_Finalize();
}
