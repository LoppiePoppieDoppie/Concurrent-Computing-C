#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

using namespace std;

int main(int argc, char *argv[]){
	int maxNumThreads = 41;
	int row = 1000;
	int column = 1000;
	int i, j, max;
	double timeStart, timeFinish;
	
	int* matrix1 = new int[row * column];
	int* minElements = new int[row];
	
	for (i = 0; i < row; i++)
		for(j = 0; j < column; j++)
			matrix1[i * column + j] = rand()%100 + 1;
		
	for (int nthreads = 1; nthreads < maxNumThreads; nthreads++){
		timeStart = omp_get_wtime();
		max = -9999999;
		
		#pragma omp parallel shared(matrix1, minElements) private(i, j) num_threads(nthreads)
		{
			#pragma omp for schedule(static)
				for (i = 0; i < row; i++)
					minElements[i] = 9999999;
			
			
			#pragma omp for schedule(static)
				for (i = 0; i < row; i++){
					minElements[i] = 9999999;
					for(j = 0; j < column; j++){
						if (matrix1[i * column + j] < minElements[i])
							minElements[i] = matrix1[i * column + j];
					}
				}
			#pragma omp for schedule(static)
				for (i = 0; i < row; i++){
					#pragma omp critical(undate)
					if (minElements[i] > max)
						max = minElements[i];
				}
				
		}
		timeFinish = omp_get_wtime();
		//cout << "\nThreads: " << nthreads << "Time: " << timeFinish - timeStart << "s" << endl;
		cout << timeFinish - timeStart << endl; // for excel
	}
	// array, made by rows consisted min elements
	//cout << "\n\n";
	//for (i = 0; i < row; i++)
		//cout << minElements[i] << endl;
	//cout << "\n\n";
	
		
	//max number af elements
	//cout << "Maximum value: " << max << endl;
		
	// output
	//printf("\n");
	//for (int i = 0; i < row; i++){
		//for (int j = 0; j < column; j++)
			//printf("%\t", matrix1[i * column + j]);
	//printf("\n");
	//}
		
	delete []matrix1;
	delete []minElements;
		
	return 0;
}
	
		