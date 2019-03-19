#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#define block 1024
#define count 1024*1024
#define DIM count*block

void func(float * a, float * b, double * result) {
	for(long i = 0; i < DIM; i++)
		*result += a[i] * b[i];
}

/*
void printVector(const char * name, float * array, int dim) {
	printf("%s = \n", name);
	for (int i = 0; i < dim; i++) {
		printf("%f ", array[i]);
	}
	printf("\n");
}
*/

int main (int argc, const char ** args) {
	
	float *a;
	float *b;
	double result = 0.0;
	float ms = 0.0;
	
	curandGenerator_t gen;
	cudaEvent_t start, end;
	
	a = (float*)calloc(sizeof(float), DIM);
	b = (float*)calloc(sizeof(float), DIM);
	
	curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);	//Create pseudo-random number generator
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);	//Set seed
	curandGenerateUniform(gen, a, DIM);		//Generate DIM floats
	curandGenerateUniform(gen, b, DIM);
	
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);	//Create an event object
	func(a, b, &result);
	cudaEventRecord(end);	//Record an event
	
	cudaEventSynchronize(end);		//Wait until event complete
	cudaEventElapsedTime(&ms, start, end);

	printf("The result of multiplication is %f\n" , result);
	printf("Time is %f msec\n", ms);
	
	/* Cleanup */
	free(a);
	free(b);
	
	curandDestroyGenerator(gen);
	return 0;
}