#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#define block 1024
#define maxsum 1024
#define count 1024*maxsum
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

__global__ void func(float * a, float * b, float * d_partsum) {
	__shared__ float mulv[block];	//Use shared memory for mulv
	
	int i = bid.x * bdim.x + tid.x;
	mulv[tid.x] = a[i]*b[i];
	
	__syncthreads();	//Wait for all warps in a block to reach that point in your code
	
	for (unsigned int stride = bdim.x >> 1; stride > 0; stride >>= 1) {
		__syncthreads();
		
		if (tid.x < stride) {
			mulv[tid.x] += mulv[tid.x + stride];
		}
	}
	
	if (tid.x == 0) {
		d_partsum[bid.x] = mulv[0];
	}
}

__global__ void vecSum(float *d_partsum,float *d_sum)
{	__shared__ float sumv[block];	//Use shared memory for sumv

	int i = bid.x * bdim.x + tid.x;
	sumv[tid.x] = d_partsum[i];
	__syncthreads();
	
	for (unsigned int stride = bdim.x >> 1; stride > 0; stride >>= 1) {
		if (tid.x < stride) {
			sumv[tid.x] += sumv[tid.x + stride];
		}
		
		__syncthreads();
		
	}
	
	if (tid.x == 0) {
		d_sum[bid.x] = sumv[0];
	}
}

__global__ void vecSumm(float *d_sum, float *res)
{	__shared__ float sumv[maxsum];	//Use shared memory for sumv

	int i = tid.x;
	sumv[tid.x] = d_sum[i];
	
	__syncthreads();
	
	for (unsigned int stride = bdim.x >> 1; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			sumv[tid.x] += sumv[tid.x + stride];
		}
		__syncthreads();
	}
	
	if (tid.x == 0) {
		*res = sumv[0];
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
	float *d_sum;
	float *d_partsum;
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
	
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_a,size));	//Allocate DIM floats on device
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_b,size));
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_result,sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_sum,sizeof(float)*maxsum));
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_partsum,sizeof(float)*count));
	
	cudaMemcpy(d_result, &res, sizeof(float), cudaMemcpyHostToDevice);		//Copies data between Host and Device
	curandGenerateUniform(gen, d_a, DIM);	//Generate DIM floats
	curandGenerateUniform(gen, d_b, DIM);

	cudaEventCreate(&start);	//Create an event object
	cudaEventCreate(&end);
	cudaEventRecord(start);		//Record an event
	func<<<count,block>>>(d_a, d_b, d_partsum);
	vecSum<<<maxsum, block>>>(d_partsum,d_sum);
	vecSumm<<<1, maxsum>>>(d_sum, d_result);
	cudaEventRecord(end);
	
	cudaEventSynchronize(end);		//Wait until event complete
	cudaMemcpy(&res, d_result, sizeof(float), cudaMemcpyDeviceToHost);		//Copies data between Device and Host
	cudaEventElapsedTime(&ms, start, end);
	printf("Time is %f msec\n", ms);
	
	/* Cleanup */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
	cudaFree(d_sum);
	cudaFree(d_partsum);
	
	curandDestroyGenerator(gen);
	return 0;
}