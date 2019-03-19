#include <stdio.h>
#include "omp.h"

int main (){
	int rank, size;
	for (int p = 1; p < 20; p++){
			printf("\n\np = %d\n", p);
	#pragma omp parallel private(rank, size) num_threads(p)
	{
			rank = omp_get_thread_num();
			size = omp_get_num_threads();
			printf("Hello from %d out of %d\n", rank, size);
	}
	}
}