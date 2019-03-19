#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>

#define MAX_NUM 100

void compare(double* a, double* b);
void merge(double* a, int left, int n, int r);
void sort(double* a, int left, int n);
void pyramid(double* a, int X_size, int size);

int main(int argc, char *argv[]){

	int rank, size;
	int X_size = 8192;
	int proc_size;

	double* X = NULL;
	double* tempX = NULL;
	double* ptrX = NULL;

	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	proc_size = X_size / size; // element quantity per proc
	X = new double[proc_size];
	
	if(rank == 0){
		tempX = new double[X_size];
		ptrX = tempX;
		
		srand(time(NULL)); 
        for(int i = 0; i < X_size; i++)
			*ptrX++ = (double)rand() / (double)(RAND_MAX) * MAX_NUM; //add with random numbers from 0.0 to 100.0

	}

	MPI_Scatter(tempX, proc_size , MPI_DOUBLE, X, proc_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	sort(X, 0, proc_size);
	MPI_Gather(X, proc_size, MPI_DOUBLE, tempX, proc_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if(rank==0){	
		pyramid(tempX, X_size, size);		
		ptrX = tempX;
        for(int i = 0; i < X_size; i++)           
            printf("%.1f\t", *ptrX++);				
	    printf("\n");				
    }
	
	delete []X;
	delete []tempX;
	
	
	MPI_Finalize();

	return 0;
}

void merge(double* a, int left, int n, int r){
	int m = r*2;
	
	if(m < n){
		merge(a, left, n, m);      
		merge(a, left + r, n, m);    
		for (int i = left + r; i + r < left + n; i += m) 
			compare(a + i, a + i + r);
	}
	else
		compare(a + left, a + left + r);
}
 //n is the length of the piece to be merged,
 //r is the distance of the elements to be compared
 
void sort(double* a, int left, int n){
	if(n > 1){
		int m = n/2;

		sort(a, left, m);
		sort(a, left + m, m);
		merge(a, left, n, 1);
	}
}

void pyramid(double* a, int X_size, int size){
	int n;
	for(int i = size / 2; i >= 1; i /= 2){
		n = X_size / i;
		for(int j = 0; j < i; j++)
			merge(a, j * n, n, 1);
	}
}

void compare(double* a, double* b){	
	if(*b < *a){
		*a = *b + *a;
		*b = *a - *b;
		*a = *a - *b;
	}
		
}
