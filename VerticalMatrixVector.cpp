#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int ProcNum = 0;      // Number of available processes 
int ProcRank = 0;     // Rank of current process

// Function for simple definition of matrix and vector elements
void DummyDataInitialization(double* pMatrix, double* pVector, int Size) {
	int i, j;  // Loop variables

	for (i = 0; i < Size; i++) {
		pVector[i] = 1;
		for (j = 0; j < Size; j++)
			pMatrix[j*Size + i] = i;
	}
}

// Function for random definition of matrix and vector elements
void RandomDataInitialization(double* pMatrix, double* pVector, int Size) {
	int i, j;  // Loop variables
	srand(unsigned(clock()));
	for (i = 0; i < Size; i++) {
		pVector[i] = rand() / double(1000);
		for (j = 0; j < Size; j++)
			pMatrix[j*Size + i] = rand() / double(1000);
	}
}

// Function for memory allocation and data initialization
void ProcessInitialization(double* &pMatrix, double* &pVector, double* &pProcVector,
	double* &pResult, double* &pProcColumns, double* &pProcResult,
	int &Size, int &ColumnNum) {
	int RestColumns; // Number of columns, that haven’t been distributed yet
	int i;             // Loop variable

	setvbuf(stdout, 0, _IONBF, 0);
	if (ProcRank == 0) {
		do {
			printf("\nEnter size of the initial objects: ");
			scanf_s("%d", &Size);
			if (Size < ProcNum) {
				printf("Size of the objects must be greater than number of processes! \n ");
			}
		} while (Size < ProcNum);
	}
	MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Determine the number of matrix rows stored on each process
	RestColumns = Size;
	for (i = 0; i < ProcRank; i++)
		RestColumns = RestColumns - RestColumns / (ProcNum - i);
	ColumnNum = RestColumns / (ProcNum - ProcRank);

	// Memory allocation
	pProcVector = new double[ColumnNum];
	pProcColumns = new double[ColumnNum * Size];
	pProcResult = new double[Size];

	// Obtain the values of initial objects elements
	if (ProcRank == 0) {
		// Initial matrix exists only on the pivot process
		pMatrix = new double[Size*Size];
		// Initial vector exists only on the pivot process
		pVector = new double[Size];
		// Result
		pResult = new double[Size];
		// Values of elements are defined only on the pivot process
		RandomDataInitialization(pMatrix, pVector, Size);
	}
}

// Function for distribution of the initial objects between the processes
void DataDistribution(double* pMatrix, double* pProcColumns, double* pVector, double* pProcVector,
	int Size, int ColumnNum) {
	int *pSendNum; // The number of elements sent to the process
	int *pSendInd; // The index of the first data element sent to the process
	int RestColumns = Size; // Number of rows, that haven’t been distributed yet
	int *pSendVecNum; // The number of elements of vector send to the process
	int *pSendVecInd; // The index of the first element in vector sent to the process

	// Alloc memory for temporary objects
	pSendInd = new int[ProcNum];
	pSendNum = new int[ProcNum];
	pSendVecInd = new int[ProcNum];
	pSendVecNum = new int[ProcNum];

	// Define the disposition of the matrix rows for current process
	ColumnNum = (Size / ProcNum);
	pSendNum[0] = ColumnNum * Size;
	pSendInd[0] = 0;
	pSendVecNum[0] = ColumnNum;
	pSendVecInd[0] = 0;
	for (int i = 1; i < ProcNum; i++) {
		RestColumns -= ColumnNum;
		ColumnNum = RestColumns / (ProcNum - i);
		pSendNum[i] = ColumnNum * Size;
		pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
		pSendVecNum[i] = ColumnNum;
		pSendVecInd[i] = pSendVecInd[i - 1] + pSendVecNum[i - 1];
	}

	// Scatter the partial vectors
	MPI_Scatterv(pVector, pSendVecNum, pSendVecInd, MPI_DOUBLE, pProcVector, pSendVecNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Scatter the columns
	MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcColumns,
		pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Free the memory
	delete[] pSendNum;
	delete[] pSendInd;
	delete[] pSendVecNum;
	delete[] pSendVecInd;
}

// Result vector replication
void ResultReplication(double* pProcResult, double* pResult, int Size) {

	// Sum all the procResult vectors into one pResult vector
	MPI_Reduce(pProcResult, pResult, Size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

}

// Function for sequential matrix-vector multiplication
void SerialResultCalculation(double* pMatrix, double* pVector, double* pResult, int Size) {
	int i, j;  // Loop variables
	for (i = 0; i < Size; i++)
		pResult[i] = 0;
	for (j = 0; j < Size; j++) {
		for (i = 0; i < Size; i++)
			pResult[i] += pMatrix[j*Size + i] * pVector[j];
	}
}

// Process rows and vector multiplication
void ParallelResultCalculation(double* pProcColumns, double* pProcVector, double* pProcResult, int Size, int ColumnNum) {
	int i, j;  // Loop variables
	for (i = 0; i < Size; i++)
		pProcResult[i] = 0;
	for (j = 0; j < ColumnNum; j++) {
		for (i = 0; i < Size; i++)
			pProcResult[i] += pProcColumns[j*Size + i] * pProcVector[j];
	}
}

// Function for formatted matrix output
void PrintMatrix(double* pMatrix, int RowCount, int ColumnCount) {
	int i, j; // Loop variables
	for (i = 0; i < RowCount; i++) {
		for (j = 0; j < ColumnCount; j++)
			printf("%7.4f ", pMatrix[j * RowCount + i]);
		printf("\n");
	}
}

// Function for formatted vector output
void PrintVector(double* pVector, int Size) {
	int i;
	for (i = 0; i < Size; i++)
		printf("%7.16f ", pVector[i]);
}

void TestDistribution(double* pMatrix, double* pVector, double* pProcVector, double* pProcColumns,
	int Size, int ColumnNum) {
	if (ProcRank == 0) {
		printf("Initial Matrix: \n");
		PrintMatrix(pMatrix, Size, Size);
		printf("Initial Vector: \n");
		PrintVector(pVector, Size);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < ProcNum; i++) {
		if (ProcRank == i) {
			printf("\nProcRank = %d \n", ProcRank);
			printf(" Matrix Stripe:\n");
			PrintMatrix(pProcColumns, Size, ColumnNum);
			printf(" Vector: \n");
			PrintVector(pProcVector, ColumnNum);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

// Fuction for testing the results of multiplication of the matrix stripe 
// by a vector
void TestPartialResults(double* pProcResult, int Size) {
	int i;    // Loop variables
	for (i = 0; i < ProcNum; i++) {
		if (ProcRank == i) {
			printf("\nProcRank = %d \n Part of result vector: \n", ProcRank);
			PrintVector(pProcResult, Size);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

// Testing the result of parallel matrix-vector multiplication
void TestResult(double* pMatrix, double* pVector, double* pResult,
	int Size) {
	// Buffer for storing the result of serial matrix-vector multiplication
	double* pSerialResult;
	// Flag, that shows wheather the vectors are identical or not
	int equal = 0;
	int i;                 // Loop variable

	if (ProcRank == 0) {
		pSerialResult = new double[Size];
		SerialResultCalculation(pMatrix, pVector, pSerialResult, Size);
		//PrintVector(pResult, Size);
		//printf("\n");
		//PrintVector(pSerialResult, Size);
		//printf("\n");
		for (i = 0; i < Size; i++) {
			if (fabs(pResult[i] - pSerialResult[i]) > 1e-6)
				equal = 1;
		}
		if (equal == 1)
			printf("The results of serial and parallel algorithms are NOT identical. Check your code.");
		else
			printf("The results of serial and parallel algorithms are identical.");

		delete[] pSerialResult;
	}
}

// Function for computational process termination
void ProcessTermination(double* pMatrix, double* pVector, double* pProcVector, double* pResult,
	double* pProcColumns, double* pProcResult) {
	if (ProcRank == 0) {
		delete[] pMatrix;
		delete[] pVector;
		delete[] pResult;
	}
	delete[] pProcVector;
	delete[] pProcColumns;
	delete[] pProcResult;
}

void main(int argc, char* argv[])
{
	double* pMatrix;  // The first argument - initial matrix
	double* pVector;  // The second argument - initial vector
	double* pProcVector; // The partial vector on the current process
	double* pResult;  // Result vector for matrix-vector multiplication 
	int Size;		    // Sizes of initial matrix and vector
	double* pProcColumns;   // Stripe of the matrix on the current process
	double* pProcResult; // Block of the result vector on the current process
	int ColumnNum;          // Number of columns in the matrix stripe
	double Duration, Start, Finish;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	if (ProcRank == 0) {
		printf("Parallel matrix-vector multiplication program\n");
	}

	// Memory allocation and data initialization
	ProcessInitialization(pMatrix, pVector, pProcVector, pResult, pProcColumns, pProcResult,
		Size, ColumnNum);

	Start = MPI_Wtime();


	// Distributing the initial objects between the processes
	DataDistribution(pMatrix, pProcColumns, pVector, pProcVector, Size, ColumnNum);

	//TestDistribution(pMatrix, pVector, pProcVector, pProcColumns, Size, ColumnNum);

	// Process rows and vector multiplication
	ParallelResultCalculation(pProcColumns, pProcVector, pProcResult, Size, ColumnNum);

	// Result replication
	ResultReplication(pProcResult, pResult, Size);

	Finish = MPI_Wtime();
	Duration = Finish - Start;

	//TestPartialResults(pProcResult, Size);
	TestResult(pMatrix, pVector, pResult, Size);

	if (ProcRank == 0) {
		printf("Time of execution = %f\n", Duration);
	}

	/*
	Size = 10000;
	pMatrix = new double[Size * Size];
	pVector = new double[Size];
	pResult = new double[Size];
	RandomDataInitialization(pMatrix, pVector, Size);
	time_t start, finish;
	start = clock();
	SerialResultCalculation(pMatrix, pVector, pResult, Size);
	finish = clock();
	double duration = (finish - start) / double(CLOCKS_PER_SEC);
	printf("\n Time of execution: %f", duration);
	*/

	// Process termination
	ProcessTermination(pMatrix, pVector, pProcVector, pResult, pProcColumns, pProcResult);

	MPI_Finalize();
}
