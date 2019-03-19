#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#define DIM 1024

void func(float *x, float *A, float *result, int dim);
void printMatrix(const char *name, float *matrix, int n, int m);
void printVector(const char *name, float *array, int dim);

int main (int argc, const char **args){
	float *A;
	float *x;
	float *multiply;
	float sum = 0;
	float ms = 0;
	
	A = (float *) malloc(DIM * DIM * sizeof(float));
	x = (float *) malloc(DIM * sizeof(float));
	multiply = (float *) malloc(DIM * sizeof(float));
	
	cudaEvent_t start, end;
	curandGenerator_t gen;

	curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10); //Create pseudo-random number generator 
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); //Set seed
	curandGenerateUniform(gen, x, DIM); //Generate DIM floats
	curandGenerateUniform(gen, A, DIM * DIM); //Generate DIM floats
	
	cudaEventCreate(&start); //Create an event object
	cudaEventCreate(&end);
	cudaEventRecord(start); //Record an event
	func(x, A, multiply, DIM);
	cudaEventRecord(end);
	
	cudaEventSynchronize(end); //Wait until event complete
	
	 for (int i = 0; i < DIM; i++)
	 sum += multiply[i];
	
	cudaEventElapsedTime(&ms, start, end);
	
	printf("time in %f msec/n", ms);
	printf("sum of all elements = %f", sum);
	
	printf("result of multiply", multiply, DIM);
	
	/* Cleanup */
    curandDestroyGenerator(gen);
	
    free(multiply); 
	free(A);
	free(x);
	
    return 0;
	
}

void func(float *x, float *A, float *result, int dim){
	for (int i = 0; i < DIM; i++){
		for (int j = 0; j < DIM; j++){
			result[i] += A[i * DIM +j] * x[j];
		}
	}
}
void printMatrix(const char *name, float *matrix, int n, int m){
	printf("%s = \n", name);
	for (int i = 0; i < DIM; i++){
		for (int j = 0; j < DIM; j++){
			printf("%f", matrix[i * m +j]);
		}
		printf("\n");
	}
}
void printVector(const char *name, float *array, int dim){
	printf("%s = \n", name);
	for (int i = 0; i < DIM; i++){
		printf("%f", array[i]);
	}
	printf("\n");
}

