#include <stdio.h>
#include "omp.h"
#include <iostream>

using namespace std;

double f(double x) {
	return 4.0 / (x*x + 1);
}

int main(int argc, char *argv[]) {

	double sum, step, timeStart, timeStop;
	
	int maxNumThreads = 41;
	int nodes_num = 100000000;
	int i;
	
	step = 1. / nodes_num;
	
	for (int nthreads=1;nthreads < maxNumThreads;nthreads++){
		timeStart = omp_get_wtime();
		sum = 0.0;
		
		#pragma omp parallel private(i) num_threads(nthreads) 
		{
			#pragma omp for schedule(static) reduction(+:sum)
				for (i = 1; i < nodes_num - 1; i += 2)
					sum += f((i-1)*step) + 4*f(i*step) + f((i+1)*step);

		}
		sum *= step/(double)3;
		timeStop = omp_get_wtime();
		//cout << "\nThreads: " << nthreads <<"  Time :   "<<timeStop - timeStart<< "  sec   "<<endl;
		cout<<timeStop - timeStart<<endl; // output for excel
	}
	
	//cout<< "\n\nPI is     : "<<sum<<endl;
	
	return 0;
}
				