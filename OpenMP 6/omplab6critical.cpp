#include <stdio.h>
#include "omp.h"
#include "vector"
#include <algorithm>
#include <stdlib.h>
#include <iostream>

#define RAND_MAX 10
using namespace std;


int main(int argc, char *argv[]) {

	int vectorSize = 1000000;
	int maxNumThreads = 41;
	int rank;
	double timeStart, timeStop, timeAver;
	
	vector<double> vectorA, vectorB;
	double multiply;
	
	for (int i = 0; i < vectorSize; i++){
		vectorA.push_back(rand()/(double)RAND_MAX+1.0);
		vectorB.push_back(rand()/(double)RAND_MAX+1.0);
	}
		
	for (int nthreads = 1; nthreads < maxNumThreads; nthreads++){
		timeStart = omp_get_wtime();
		
		omp_lock_t lock;
		omp_init_lock(&lock); 
		
		#pragma omp parallel num_threads(nthreads) 
		{
			rank = omp_get_thread_num();
			multiply = 0.0;
		
			for (int i = rank; i < vectorSize; i += nthreads)
				{
					#pragma omp critical(Section1)
					multiply += vectorA[i] * vectorB[i];
				}	

		timeStop = omp_get_wtime();
		}
		omp_destroy_lock (&lock); 
		//cout << "\nThreads: " << nthreads <<"  Time :   "<<timeStop - timeStart<< "  sec   "<<endl;
		cout<<timeStop - timeStart<<endl; // output for excel
		timeAver += timeStop - timeStart;
	}
	timeAver = timeAver / (maxNumThreads-1);
	//cout << "\nAverage Time: " << timeAver<<"  sec  "<<endl;
		
	cout <<"\n\nMultiplication :"<< multiply<< "\n\n";
	
	return 0;
}