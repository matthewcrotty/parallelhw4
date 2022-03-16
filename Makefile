all: reduce-mpi.c reduce-cuda.cu
	mpicc -g reduce-mpi.c -c -o reduce-mpi.o
	nvcc -g -G -arch=sm_70 reduce-cuda.cu -c -o reduce-cuda.o
	mpicc -g reduce-mpi.o reduce-cuda.o -o reduce-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++
