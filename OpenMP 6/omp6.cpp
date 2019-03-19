#include <stdio.h>
#include "omp.h"
#include "vector"
#include <algorithm>
#include <stdlib.h>
#include <iostream>

#define RAND_MAX 10
using namespace std;


int main(int argc, char *argv[]) {

	int vectorSize = 100000000;
	int maxNumThreads = 41;
	int rank;
	double time1, time2, timeav;
	
	vector<double> vectorA, vectorB;
	double multip;
	
	for (int i = 0;i < vectorSize; i++){
		vectorA.push_back(rand()/(double)RAND_MAX+1.0);
		vectorB.push_back(rand()/(double)RAND_MAX+1.0);
	}
	
	/*cout <<"\n";
	cout <<"VectorA : ";
	for(int i = 0;i < vectorSize;i++)
		cout<<vectorA[i]<<"  ";
	cout <<endl<<"VectorB : ";	
	for(int i = 0;i < vectorSize;i++)
		cout<<vectorB[i]<<"  ";
	cout <<"\n";*/
	
	for (int nthreads=1;nthreads < maxNumThreads;nthreads++){
		time1 = omp_get_wtime();
		
		omp_lock_t lock;
		omp_init_lock(&lock); 
		
		#pragma omp parallel num_threads(nthreads) 
		{
			rank = omp_get_thread_num();
			multip = 0.0;
		
			//-----------------------------------------------
			//Reduction method
			/*#pragma omp for schedule(static) reduction(+:multip)
				for(int i = 0; i < vectorSize; i++)
					multip += vectorA[i] * vectorB[i];*/
			//-----------------------------------------------
			
			//Atomic method 
				/*for(int i = rank; i < vectorSize; i += nthreads)
				{
					#pragma omp atomic
					multip += vectorA[i] * vectorB[i];
				}*/
			//-----------------------------------------------
				
			//Critical sections method
				/*for(int i = rank; i < vectorSize; i += nthreads)
				{
					#pragma omp critical(Section1)
					multip += vectorA[i] * vectorB[i];
				}*/
			//-----------------------------------------------
			
			//Synchronization/block method
				#pragma omp for 
				for(int i = 0; i < vectorSize; i++)
				{
					omp_set_lock (&lock);
					multip += vectorA[i] * vectorB[i];
					omp_unset_lock (&lock); 
				}
			//-----------------------------------------------

		time2 = omp_get_wtime();
		}
		omp_destroy_lock (&lock); 
		//cout << "\nThreads: " << nthreads <<"  Time :   "<<time2 - time1<< "  sec   "<<endl;
		cout<<time2 - time1<<endl; // output for excel
		timeav += time2 - time1;
	}
	timeav = timeav / (maxNumThreads-1);
	//cout << "\nAverage Time: " << timeav<<"  sec  "<<endl;
		
	cout <<"\n\nMultiplication :"<<multip<<"\n\n";
	
	return 0;
}