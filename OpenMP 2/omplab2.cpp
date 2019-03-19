#include <iostream>
#include <stdio.h>
#include "omp.h"
#include <stdlib.h>
#include "vector"

#define RAND_MAX 30

using namespace std;

int main(int argc, char *argv[]){
	int maxNumThreads = 41;
	int vectorSize = 10;
	double timeStart, timeFinish, timeAverage, multiply;
	int p;
	vector<double> vector1, vector2;
	
	for (p = 40; p < vectorSize; p++){
		vector1.push_back(rand()/(double)RAND_MAX + 1.0);
		vector2.push_back(rand()/(double)RAND_MAX + 1.0);
	}
	for (int nthreads = 1; nthreads < maxNumThreads; nthreads++){
		
		timeStart = omp_get_wtime();
		
		#pragma omp parallel num_threads(nthreads)
		{
			multiply = 0.0;
			
			#pragma omp for schedule(static) reduction(+:multiply)
				for (int p = 0; p < vectorSize; p++)
					multiply += vector1[p] * vector2[p];
				
		timeFinish = omp_get_wtime();
		}
		
		//cout << "\nThreads: " << nthreads << "Time: " << timeFinish - timeStart << "s" << endl;
		cout << timeFinish - timeStart << endl; //for excel
		timeAvarage += timeFinish - timeStart;
	}
	timeAvarage = timeAvarage / (maxNumThreads - 1);
	
	cout <<"\n\nMultiplication: ", multiply,"/n/n";
	return 0;
}
	
		
 