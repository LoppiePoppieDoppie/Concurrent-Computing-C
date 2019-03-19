#include <stdio.h>
#include "omp.h"
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {

	int matrixSize = 500;
	int maxNumThreads = 41;
	int i, j, k;
	double timeStart, timeStop, timeAver;
	
	int* matrixA = new int[matrixSize * matrixSize];
	int* matrixB = new int[matrixSize * matrixSize];
	int* matrixC = new int[matrixSize * matrixSize];
	
	for(i = 0; i < matrixSize; i++)
		for(j = 0; j < matrixSize; j++)
			matrixA[i * matrixSize + j] = i + j;
	
	for(i = 0; i < matrixSize; i++)
		for(j = 0; j < matrixSize; j++)
			matrixB[i * matrixSize + j] = i * j;
			
	timeAver = 0.0;
			
	for (int nthreads=1;nthreads < maxNumThreads;nthreads++){
		timeStart = omp_get_wtime();

		#pragma omp parallel shared(matrixA, matrixB, matrixC) private(i, j, k) num_threads(nthreads) 
		{
			#pragma omp for schedule(static)
				for(i = 0; i < matrixSize; i++){
					for(j = 0; j < matrixSize; j++){
						matrixC[i * matrixSize + j] = 0;
						for(k = 0; k < matrixSize; k++)
							matrixC[i * matrixSize + j] += matrixA[i * matrixSize + k] * matrixB[j * matrixSize + k];
					}
				}
		}
		timeStop = omp_get_wtime();
		//cout << "\nThreads: " << nthreads <<"  Time :   "<<timeStop - timeStart<< "  sec   "<<endl;
		cout<<timeStop - timeStart<<endl; // output for excel
		timeAver+= timeStop - timeStart;
	}
	
	timeAver = timeAver / (maxNumThreads-1);
	//cout << "\nAverage Time: " << timeAver <<"  sec  "<<endl;
	
	/* clean up*/
	delete []matrixA;
	delete []matrixB;
	delete []matrixC;
	
	return 0;
}