#include <stdio.h>

extern "C" double* input_data;


struct SharedMemory {
    __device__ inline operator double *() {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};


__device__ __forceinline__ double warpReduceSum(unsigned int mask, double mySum) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        mySum += __shfl_down_sync(mask, mySum, offset);
    }
    return mySum;
}


template <unsigned int blockSize>
__global__ void reduce7(const double *__restrict__ g_idata, double *__restrict__ g_odata,
            unsigned int n) {
    double *sdata = SharedMemory();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize * gridDim.x;
    unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
    maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
    const unsigned int mask = (0xffffffff) >> maskLength;

    double mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread

    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
        mySum += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for
        // powerOf2 sized arrays
        if ((i + blockSize) < n) {
            mySum += g_idata[i + blockSize];
        }
        i += gridSize;
    }

    // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum(mask, mySum);

    // each thread puts its local sum into shared memory
    if ((tid % warpSize) == 0) {
        sdata[tid / warpSize] = mySum;
    }

    __syncthreads();

    const unsigned int shmem_extent = (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
    const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
    if (tid < shmem_extent) {
        mySum = sdata[tid];
        // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
        // SM 8.0
        mySum = warpReduceSum(ballot_result, mySum);
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}


extern "C" void initCuda(int my_rank, int num_elements, double** data){
    int cudaDeviceCount = 0;
    cudaError_t cE;
    if( (cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess){
        printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount);
        exit(-1);
    }

    if( (cE = cudaSetDevice(my_rank % cudaDeviceCount)) != cudaSuccess){
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n", my_rank, (my_rank % cudaDeviceCount), cE);
        exit(-1);
    }

    printf("Mapping Rank %d to CUDA Device %d \n", my_rank, (my_rank % cudaDeviceCount));

    cudaMallocManaged(&input_data, num_elements * sizeof(double));
    for(int i = 0; i < num_elements; i++){
        input_data[i] = i + (num_elements * my_rank);
    }

}

extern "C" void reduceCuda(int num_elements, int threads, int blocks, double* input, double* output, int my_rank){
    for(int i = 0; i < num_elements; i++){
        printf("Rank %d, %f\n", my_rank, input[i]);
    }
}
