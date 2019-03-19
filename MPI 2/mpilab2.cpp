#include <iostream>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]){
	int rank, proc, size;
	MPI_Status status; 
	char buf[64];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	sprintf(buf, "Hello from process %d", rank);
	
	proc = (rank + 1) % 10;
	MPI_Send(buf, 64, MPI_CHAR, proc, 0, MPI_COMM_WORLD);
	proc = (rank + 9) % 10;
	MPI_Recv(buf, 64, MPI_CHAR, proc, 0, MPI_COMM_WORLD, &status);
	
	printf("Process %d received from %d: %s, size, %d\n", rank, proc, buf, size);
	
	MPI_Finalize();
	
	return 0;
}
