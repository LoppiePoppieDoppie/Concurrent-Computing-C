#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char *argv[]){
	int rank, size;
	double* A;
	double* x;
	double tempX, f, norm, maxNorm, eps;
	
	MPI_Init(&argc, &argv);

        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	A = new double[size];
	x = new double[size];
	
	maxNorm = -1;
	eps = 0.001;
	
	for(int i = 0; i < size; i++){
		x[i] = -1;
		A[i] = 1;
	}
	
	A[rank] = 2 * size;
	f = size * (size + 1) / 2 + (rank+1) * (2 * size - 1);
	
	do{
		tempX = f;
		for(int i = 0; i < size; i++){
			if(i != rank)
				tempX -= A[i] * x[i];
		}
		
		tempX /= A[rank];
		norm = fabs(tempX - x[rank]);
		
		MPI_Allgather(&tempX, 1, MPI_DOUBLE, x, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Reduce(&norm,&maxNorm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Bcast(&maxNorm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		while(maxNorm > eps);
	
	if(rank==0)
		for(int i = 0; i < size; i++)
			printf("x[%d] = %f\n", i, x[i]);
			
	delete []A;
	delete []x;
	
	MPI_Finalize();

	return 0;
}