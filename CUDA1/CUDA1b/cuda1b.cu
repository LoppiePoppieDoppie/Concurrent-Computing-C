#include "stdio.h"
#include "cuda.h"
#include "curand.h"
#include <curand_kernel.h>
#define DIM 1024

__global__ void func(float * x, float * A, float * result, int dim) {
	int i = threadIdx.x;
		for (int j = 0; j < dim; j++) {
			result[i] += A[i * dim + j] * x[j];
		}
}

void printMatrix(const char * name, float * matrix, int n, int m) {
	printf("%s = \n", name);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			printf("%f ", matrix[i * m + j]);
		}
		printf("\n");
	}
}

void printVector(const char * name, float * array, int dim) {
	printf("%s = \n", name);
	for (int i = 0; i < dim; i++) {
		printf("%f ", array[i]);
	}
	printf("\n");
}

void zeros(float * array, int dim) {
	for (int i = 0; i < dim; i++) {
		array[i] = 0.;
	}
}

int main (int argc, const char ** args) {
	float *mul;
	float *d_x;
	float *d_A;
	float *d_result;
	float ms = 0;
	float sum = 0;
	
	curandGenerator_t gen;
	cudaEvent_t start, end;
	
	mul = (float *) malloc(DIM * sizeof(float));
	cudaMalloc((void **) &d_x, DIM * sizeof(float));	//Allocate DIM floats on device
	cudaMalloc((void **) &d_A, DIM * DIM * sizeof(float));
	cudaMalloc((void **) &d_result, DIM * sizeof(float));
	
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);	//Create pseudo-random number generator 
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);	//Set seed
	curandGenerateUniform(gen, d_x, DIM);	//Generate DIM floats
	curandGenerateUniform(gen, d_A, DIM * DIM);	//Generate DIM floats
	
	zeros(mul, DIM);	
	cudaMemcpy(d_result, mul, DIM * sizeof(float), cudaMemcpyHostToDevice);	//Copies data between Host and Device
		
	cudaEventCreate(&start);	//Create an event object
	cudaEventCreate(&end);	
	cudaEventRecord(start);	//Record an event
	func<<<1, DIM>>>(d_x, d_A, d_result, DIM);
	cudaEventRecord(end);
	
	cudaMemcpy(mul, d_result, DIM * sizeof(float), cudaMemcpyDeviceToHost);	//Copies data between Device and Host

	cudaEventSynchronize(end);	//Wait until event complete
	cudaEventElapsedTime(&ms, start, end);

	for(int i=0;i<DIM;i++)
	sum += mul[i];
		
	printf("Time is %f msec\n", ms);
	printf("Sum of elements = %f", sum);
	
	/* Cleanup */
	free(mul);
	
	cudaFree(d_x);
	cudaFree(d_A);
	cudaFree(d_result);
	
	curandDestroyGenerator(gen);
	return 0;
}