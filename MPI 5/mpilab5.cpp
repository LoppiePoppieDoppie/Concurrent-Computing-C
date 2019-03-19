#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <math.h>


int main(int argc, char *argv[]){
	int size, rank, bound, rowCount, maxIter, currentIter, boundaryIter;
	double norm, maxNorm, eps;
	
	double* A = NULL;
	double* B = NULL;
	double* tempA = NULL;
	double* ptrA = NULL;
	double* ptrB = NULL;
	
	int Asize = 1000;
	int* sendct = NULL;
	int* shift = NULL;
	int* ptrSend = NULL;
	int* ptrShift = NULL;


	MPI_Status status;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	

	bound = (rank%(size-1))==0;
	rowCount = Asize / size + 2 - bound;
	
	eps = 0.001;
	currentIter = 0;
	boundaryIter = 2500;

	
	if (rank == 0) {
	printf("\nDomain size is %d*%d, number of threads is %d\n", Asize, Asize, size);
	
	tempA = new double[Asize * Asize];
	sendct = new int[size];
	shift = new int[size];
	ptrSend = sendct;
	ptrShift = shift;
	ptrA = tempA;
	
	for(int i=0;i<Asize;i++)
		*ptrA++ = 1;
		
	for(int i=1;i<Asize - 1;i++){
		*ptrA++ = 1;
			for(int j=1;j<Asize - 1;j++)
				*ptrA++ = 0;
			*ptrA++ = 1;
		}
	
	for(int i=0;i<Asize;i++)
		*ptrA++ = 1;
		
	*ptrSend++ = rowCount * Asize;
	*ptrShift++ = 0;
	*ptrShift = (rowCount - 2) * Asize;
	
	for(int i = 1; i <  size - 1; i++){
			*ptrSend++ = (rowCount + 1) * Asize;
			*++ptrShift = shift[1] + i * (rowCount - 1) * Asize;
		}
		*ptrSend = rowCount * Asize;
		printf("\n%d iterations to go\n\n\n", boundaryIter);
	}
	
	A = new double[rowCount * Asize];
	B = new double[(Asize-2) * (rowCount-2)];
	MPI_Scatterv(tempA, sendct, shift, MPI_DOUBLE, A, rowCount * Asize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	
	do{
		maxNorm = -1.0;
		ptrB = B;
		ptrA = A + Asize + 1;
		for(int i = 0; i < rowCount - 2; i++){
			for(int j = 0; j < Asize - 2; j++){
				*ptrB = 1.0/4 * (*(ptrA-1) + *(ptrA + 1) + *(ptrA-Asize) + *(ptrA+Asize));
				norm = fabs(*ptrB - *ptrA);
				if(norm > maxNorm)
					maxNorm = norm;
				ptrB++;
				ptrA++;
			}
			ptrA += 2;
		}
						
		ptrB = B;
		ptrA = A + Asize + 1;
		for(int i = 0; i < rowCount - 2; i++){
			for(int j = 0; j < Asize - 2; j++)
				*ptrA++ = *ptrB++;
			ptrA += 2;
		}
		
	
		if(rank < size - 1) 
			MPI_Send(B + (Asize-2) * (rowCount-3), Asize-2, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD); 
		if(rank > 0) 
			MPI_Recv(A + 1, Asize-2, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &status); 
		if(rank > 0) 
			MPI_Send(B, Asize-2, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
		if(rank < size - 1) 
			MPI_Recv(A + Asize * (rowCount-1) + 1, Asize-2, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &status);
			
		MPI_Reduce(&maxNorm, &norm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Bcast(&norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		
		currentIter++;
		if(!(currentIter % 50) && (rank == 0)){
           printf("%d iterations left, T(20,20) is %.3f\n", currentIter, *(A + 20*Asize + 20));
        }
	
	}while(norm > eps && currentIter < boundaryIter);
	
	MPI_Gather(A + (rank!=0) * Asize, Asize *	Asize / size, MPI_DOUBLE, tempA, Asize * Asize / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if(rank==0){
		ptrA = tempA;
		
        if(currentIter==boundaryIter)	
			printf("\n\nSolution not found. \nCurrent error is %.5f. Target error is %.5f\n", boundaryIter, norm, eps);
		else			
			printf("\n\nSolution found and required %d iterations.\nCurrent error is %.5f. Target error is %.5f\n", currentIter, norm, eps);
    }
	
	delete []A;
	delete []B;
	delete []tempA;
	delete []sendct;
	delete []shift;
	
	
	MPI_Finalize();

	return 0;
}		