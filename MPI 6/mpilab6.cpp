#include <iostream>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]){

	int* A = NULL;
	int* B = NULL;
	int* C = NULL;
	int* tempB = NULL;
	int* tempC = NULL;
	int* sendct = NULL;
	int* shift = NULL;

	int A_row = 100, A_column = 15;
	int B_row = 15, B_column = 7;
	int A_size, B_size, C_size, B_shift, Brow, Bcolumn, size, rank;
	int mgsize;
		
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	mgsize = B_row * B_column / size;
	
	A_size = A_row * A_column;
	B_size = B_row * B_column;
	C_size = A_row * B_column;
	
	A = new int[A_size];
	tempB = new int[B_size];
	tempC = new int[C_size];
	
	if(rank == 0) {
		B = new int[B_size];
		C = new int[C_size];
		
		sendct = new int[size];
		shift = new int[size];
	
		for(int i=0; i<A_row; i++)
		  for(int j=0; j<A_column; j++)
			A[i * A_column + j] = i + j;

		for (int i=0; i<B_row; i++)
		  for(int j=0; j<B_column; j++)
			B[i * B_column + j] = i*j;
			
		for(int i=0; i<C_size;i++){
			tempC[i] = 0;
			C[i] = 0;
		}
		
		for(int i = 0; i < size - 1; i++){
            sendct[i] = mgsize;
            shift[i] = i * mgsize;
		}
		
		sendct[size - 1] = B_size - (size - 1) * mgsize;
		shift[size - 1] = (size - 1) * mgsize;		
	}

	MPI_Bcast(A, A_row * A_column, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tempC, C_size , MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(sendct, 1, MPI_INT, &B_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(shift, 1, MPI_INT, &B_shift, 1, MPI_INT, 0, MPI_COMM_WORLD);

	tempB = new int[B_size];
	MPI_Scatterv(B, sendct, shift, MPI_INT, tempB, B_size, MPI_INT, 0, MPI_COMM_WORLD);
	
	for(int i = 0; i < B_size; i++){
        Brow = (B_shift + i) / B_column;
        Bcolumn = (B_shift + i) % B_column;
        for(int j = 0; j < A_row; j++)
            tempC[j * B_column + Bcolumn] +=  A[j * A_column + Brow] * tempB[i];
    }
	
	MPI_Reduce(tempC, C, C_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	
    if(rank==0){
        printf("C = \n");
        for(int i = 0; i < A_row; i++){
            for(int j = 0; j < B_column; j++)
                printf("%d\t", C[i * B_column + j]);				
            printf("\n");
        }		
	}
	
	delete []A;
	delete []B;
	delete []C;
	delete []tempB;
	delete []tempC;
	delete []sendct;
	delete []shift;
	
	MPI_Finalize();
	return 0;
}