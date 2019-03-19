#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#define block 1024
#define count 1024*1024
#define DIM count*block
#define tid threadIdx
#define bid blockIdx
#define bdim blockDim

#define CUDA_DEBUG

#ifdef CUDA_DEBUG

#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}                 \

#else

#define CUDA_CHECK_ERROR(err)

#endif

__global__ void func(float * a, float * b, float * result) {
	__shared__ float mulv[block];	//Use shared memory for mulv
	
	int i = bid.x * bdim.x + tid.x;
	mulv[tid.x] = a[i] * b[i];
	__syncthreads();	//Wait for all warps in a block to reach that point in your code
	
	if(tid.x == 0){
		float res = 0;
		
		for(int i = 0;i < block; i++){
			res += mulv[i];
		}
		atomicAdd(result, res);
	}
}

void printVector(const char * name, float * array, int dim) {
	printf("%s = \n", name);
	for (int i = 0; i < dim; i++) {
		printf("%f ", array[i]);
	}
	printf("\n");
}
int main (int argc, const char ** args) {
	
	float *d_a;
	float *d_b;
	float *d_result;
	float res = 0.0;
	float ms = 0.0;
	
	curandGenerator_t gen;
	cudaEvent_t start, end;

	unsigned long size = sizeof(float) * DIM;
	size_t heapSize = size * 3;
	
	/* Set resource limits */
	CUDA_CHECK_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));	//GPU malloc heap size
	
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);	//Create pseudo-random number generator 
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);	//Set seed
	
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_a,size));		//Allocate DIM floats on device
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_b,size));
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_result,sizeof(float)));
	
	cudaMemcpy(d_result, &res, sizeof(float), cudaMemcpyHostToDevice);		//Copies data between Host and Device
	
	curandGenerateUniform(gen, d_a, DIM);	//Generate DIM floats
	curandGenerateUniform(gen, d_b, DIM);
	
	cudaEventCreate(&start);	//Create an event object
	cudaEventCreate(&end);
	cudaEventRecord(start);		//Record an event
	func<<<count,block>>>(d_a, d_b, d_result);
	cudaEventRecord(end);
	
	cudaEventSynchronize(end);		//Wait until event complete
	cudaMemcpy(&res, d_result, sizeof(float), cudaMemcpyDeviceToHost);		//Copies data between Device and Host
	cudaEventElapsedTime(&ms, start, end);
	
	printf("Time is %f msec\n", ms);
	
	/* Cleanup */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
	
	curandDestroyGenerator(gen);
	return 0;
}