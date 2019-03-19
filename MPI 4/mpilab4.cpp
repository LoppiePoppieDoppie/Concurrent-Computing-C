#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]){
	int maxRandL = 1000000;
	int rank, size, i, L, sluc[maxRandL], send, comm;

	double timeStart, timeFinish;
	double timeDifference = 0;
	double timeAver = 0;
	
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	for (i = 0; i < maxRandL; i++){
		sluc[i] = rand();
	}
		for (L = 1; L <= maxRandL; L *= 10){
			timeAver = 0;
			for (i = 1; i <= 100; i++){
				timeStart = MPI_Wtime();
				if (rank == 0){
					int send[L];
					for (comm = 0; comm < L; comm++)
						send[comm] = sluc[comm];
						MPI_Send(&send, L, MPI_INT, (rank + 1), 1, MPI_COMM_WORLD);
						MPI_Recv(&send, L, MPI_INT, (rank + 1), 1, MPI_COMM_WORLD, &status);
				
				}
				if (rank == 1){
					int recv[L];
					MPI_Recv(&recv, L, MPI_INT, (rank - 1), 1, MPI_COMM_WORLD, &status);
					MPI_Send(&recv, L, MPI_INT, (rank - 1), 1, MPI_COMM_WORLD);
					timeFinish = MPI_Wtime();
					timeDifference = timeFinish - timeStart;
					
					timeAver += timeDifference;
				}
			}
			if (rank == 1){
				timeAver = timeAver/100;
				double speed = 32*L/(timeAver*1024*1024);
				printf("for %d that took %f sec, speed = %f mBs\n", L, timeAver, speed);
			}
		}
	MPI_Finalize();
	return 0;
}