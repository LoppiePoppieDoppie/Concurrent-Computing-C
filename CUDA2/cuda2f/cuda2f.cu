#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#define block 1024
#define block_part 32
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

__inline__ __device__
float warpReduceSum(float res) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    res += __shfl_down_sync(0xffffffff,res, offset);	//exchange a variable between threads within a warp and get res from lane offset
  return res;
}

__inline__ __device__
float blockReduceSum(float res, float *sumv) {
	int lane = tid.x % warpSize;
	int wid = tid.x / warpSize;

	res = warpReduceSum(res);

	if (lane==0) sumv[wid]=res; 

	__syncthreads(); 
	if(tid.x < bdim.x / warpSize)
	{
		res = warpReduceSum(sumv[lane]);
	}

	return res;
}

__global__ void func(float * a, float * b, float * sum) {
	__shared__ float sumv[block_part];
	
	int i = bid.x * bdim.x + tid.x;
	float res = blockReduceSum(a[i]*b[i],sumv);

	if (tid.x == 0) 
	{
		sum[bid.x] = res;
	}
}

__global__ void vecSum(float *partsum,float *sum)
{	__shared__ float sumv[block_part];

	int i = bid.x * bdim.x + tid.x;
	float res = blockReduceSum(partsum[i],sumv);

	if (tid.x == 0) 
	{
		sum[bid.x] = res;
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
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_sum,sizeof(float)*count));
	CUDA_CHECK_ERROR(cudaMalloc((void **) &d_partsum,sizeof(float)*count));
	
	cudaMemcpy(d_result, &res, sizeof(float), cudaMemcpyHostToDevice);		//Copies data between Host and Device
	curandGenerateUniform(gen, d_a, DIM);	//Generate DIM floats
	curandGenerateUniform(gen, d_b, DIM);

	cudaEventCreate(&start);	//Create an event object
	cudaEventCreate(&end);
	cudaEventRecord(start);		//Record an event
	func<<<count, block>>>(d_a, d_b, d_partsum);
	vecSum<<<count / block, block>>>(d_partsum,d_sum);
	vecSum<<<1, block>>>(d_sum, d_result);
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