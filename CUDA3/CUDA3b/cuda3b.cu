#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#define DIM 4096
#define block 32

__global__ void func(float * A, float * A_tr) {
	int x = blockIdx.x * block + threadIdx.x;
	int y = blockIdx.y * block + threadIdx.y;
	
	int i = gridDim.x * block;
	int j = gridDim.y * block;
	
	A_tr[x * j + y] = A[y * i + x];
}

int main (int argc, const char ** args) {
	
	float *d_A;
	float *d_A_tr;
	float ms = 0.0;
	
	curandGenerator_t gen;
	cudaEvent_t start, end;

	cudaMalloc((void **) &d_A, DIM * DIM * sizeof(float));		//Allocate DIM floats on device
	cudaMalloc((void **) &d_A_tr, DIM * DIM * sizeof(float));
	
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);	//Create pseudo-random number generator 
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);	//Set seed
	curandGenerateUniform(gen, d_A, DIM * DIM);		//Generate DIM floats
	
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	/*kernel config*/
	dim3 threads(block, block);		//block size
    dim3 grid(DIM / threads.x, DIM / threads.y);
	
	cudaEventRecord(start);		//Create an event object
	func<<<grid,threads>>>(d_A, d_A_tr);
	cudaEventRecord(end);	//Record an event
	
	cudaEventSynchronize(end);	//Wait until event complete
	cudaEventElapsedTime(&ms, start, end);
	printf("Time is %f msec\n", ms);
	//printf("Sum of elements = %f\n\n", sum);
	/* Cleanup */
	cudaFree(d_A);
	cudaFree(d_A_tr);
	
	curandDestroyGenerator(gen);
	return 0;
}
