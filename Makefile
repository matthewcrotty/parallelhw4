all: reduce-mpi.c reduce-cuda.cu
	mpixlc -g reduce-mpi.c -c -o reduce-mpi.o
	nvcc -g -G -arch=sm_70 reduce-cuda.cu -c -o reduce-cuda.o
	mpixlc -g reduce-mpi.o reduce-cuda.o -o reduce-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++
