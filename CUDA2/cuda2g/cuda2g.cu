#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>

#define block 1024
#define count 1024*1024
#define DIM count*block

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
	float res = 0.0;
	float ms = 0.0;
	
	curandGenerator_t gen;
	cudaEvent_t start, end;

	unsigned long size = sizeof(float) * DIM;
	size_t heapSize = size * 3;
	
	CUDA_CHECK_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));
	
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);	//Create pseudo-random number generator
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);	//Set seed
	
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_a,size));	//Allocate floats on device
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_b,size));
	
	curandGenerateUniform(gen, d_a, DIM);	//Generate DIM floats
	curandGenerateUniform(gen, d_b, DIM);
	
	thrust::device_ptr<float> dev_a(d_a);	// dev_a has storage for d_a floats
	thrust::device_ptr<float> dev_b(d_b);	// dev_b has storage for d_b floats
	
	cudaEventCreate(&start);	//Create an event object
	cudaEventCreate(&end);
	cudaEventRecord(start);		//Record an event
	res = thrust::inner_product(dev_a, dev_a + DIM, dev_b, 0.0f); 
	cudaEventRecord(end);		
	
	cudaEventSynchronize(end);		//Wait until event complete
	cudaEventElapsedTime(&ms, start, end);
	printf("Time is %f msec\n", ms);
	printf("The result of multiplication is %lf", res);
	
	/* Cleanup */
	cudaFree(d_a);
	cudaFree(d_b);
	
	curandDestroyGenerator(gen);
	return 0;
}