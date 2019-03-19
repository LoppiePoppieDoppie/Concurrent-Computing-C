#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#define DIM 4096

void func(float * A, float * Atr) {
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < DIM; j++) {
			Atr[j * DIM + i] = A[i * DIM + j];
		}
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

int main (int argc, const char ** args) {
	
	float *A;
	float *Atr;
	float ms = 0.0;

	curandGenerator_t gen;
	cudaEvent_t start, end;
	
	A = (float *) malloc(DIM * DIM * sizeof(float));	//Allocate DIM floats on device
	Atr = (float *) malloc(DIM * DIM * sizeof(float));
	
	curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);	//Create pseudo-random number generator
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);	//Set seed
	curandGenerateUniform(gen, A, DIM * DIM);	//Generate DIM floats
	
	cudaEventCreate(&start);	//Create an event object
	cudaEventCreate(&end);
	cudaEventRecord(start);		//Record an event
	func(A, Atr);
	cudaEventRecord(end);
	
	cudaEventSynchronize(end);		//Wait until event complete
	cudaEventElapsedTime(&ms, start, end);
	
	printf("Time is %f msec\n", ms);
	//printf("Sum of elements = %f\n\n", sum);
	/* Cleanup */
	free(A);
	free(Atr);
	
	curandDestroyGenerator(gen);
	return 0;
}