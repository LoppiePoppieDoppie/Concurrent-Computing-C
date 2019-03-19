#include "stdio.h"
#include "cuda.h"
#include "curand.h"

#define DIM 4096
#define block 32

__global__ void func(float * A, float * A_tr) {
	__shared__ float temp[block][block + 1];
	
	int Ind_x = blockIdx.x * blockDim.x + threadIdx.x;
	int Ind_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	int i = gridDim.x * block;
	int j = gridDim.y * block;
	
	temp[threadIdx.y][threadIdx.x] = A[Ind_y * i + Ind_x];
	__syncthreads();	//Wait for all warps in a block to reach that point in your code
	
	Ind_x = blockIdx.y * blockDim.y + threadIdx.x;
	Ind_y = blockIdx.x * blockDim.x + threadIdx.y;
	
	A_tr[Ind_y * j + Ind_x] = temp[threadIdx.x][threadIdx.y];
}

int main (int argc, const char ** args) {
	
	float *d_A;
	float *d_A_tr;
	float ms = 0.0;
	
	curandGenerator_t gen;
	cudaEvent_t start, stop;

	cudaMalloc((void **) &d_A, DIM * DIM * sizeof(float));	//Allocate DIM floats on device
	cudaMalloc((void **) &d_A_tr, DIM * DIM * sizeof(float));
	
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);	//Create pseudo-random number generator
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);	//Set seed
	curandGenerateUniform(gen, d_A, DIM * DIM);		//Generate DIM floats
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	/*kernel config*/
	dim3 threads(block, block);
    dim3 grid(DIM / threads.x, DIM / threads.y);
	
	cudaEventRecord(start);	//Create an event object
	func<<<grid,threads>>>(d_A, d_A_tr);
	cudaEventRecord(stop);	//Record an event
	
	cudaEventSynchronize(stop);		//Wait until event complete
	cudaEventElapsedTime(&ms, start, stop);
	printf("Time is %f msec\n", ms);
	//printf("Sum of elements = %f\n\n", sum);
	/* Cleanup */
	cudaFree(d_A);
	cudaFree(d_A_tr);
	
	curandDestroyGenerator(gen);
	return 0;
}
