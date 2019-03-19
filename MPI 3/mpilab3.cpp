#include <iostream>
#include <stdio.h>
#include <mpi.h>

double func(double x){
	return 4./(x*x +1);
}

int main(int argc, char*argv[]){
	int rank, size, i;
	int n = 1000000;
	
	double pi, term, h, startTime, stopTime;
	double sum = 0.0;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	h = 1.0/n;
	startTime = MPI_Wtime();
	for(i = rank+1; i<=n; i+=size)
		if(i == 0||i == n)
			sum += func(h*(i));
		else
			if(i%2 == 1)
				sum += 4*func(h*(i));
			else 
				sum += 2*func(h*(i));
	term = h*sum/3.0;
	MPI_Reduce(&term, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	stopTime = MPI_Wtime();
	
	if (rank == 0) {
		printf("Pi = %lg\n", pi);
		printf("Time is %f sec\n", stopTime-startTime);
	}
	MPI_Finalize();
	
	return 0;
}
	