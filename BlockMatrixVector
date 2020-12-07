#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int ProcNum = 0;      // Number of available processes 
int ProcRank = 0;     // Rank of current process

void PrintVector(double* pVector, int Size);
void PrintMatrix(double* pMatrix, int RowNum, int ColumnNum);

int determineMaxWidth(int Size) {
	for (int i = sqrt(Size); i >= 1; i--)
		if (Size % i == 0)
			return i;
}

// Function for simple definition of matrix and vector elements
void DummyDataInitialization(double* original, double* pMatrix, double* pVector, int Size, int h, int w) {
	int i, j;  // Loop variables

	for (i = 0; i < Size; i++) {
		pVector[i] = 1;
		for (j = 0; j < Size; j++) {
			int ind = (i - (i % h)) * Size + (j - (j % w)) * h + (i % h) * w + (j % w);
			pMatrix[ind] = i;
			original[i * Size + j] = i;

		}
	}
	PrintMatrix(original, Size, Size); // show the original input matrix
	PrintMatrix(pMatrix, Size, Size); // show the transformed matrix 
}

// Function for random definition of matrix and vector elements
void RandomDataInitialization(double* original, double* pMatrix, double* pVector, int Size, int h, int w) {
	int i, j;  // Loop variables
	srand(unsigned(clock()));
	for (i = 0; i < Size; i++) {
		pVector[i] = rand() / double(1000);
		for (j = 0; j < Size; j++) {
			double randVal = rand() / double(1000);
			int ind = (i - (i % h)) * Size + (j - (j % w)) * h + (i % h) * w + (j % w);
			pMatrix[ind] = randVal;
			original[i * Size + j] = randVal;
		}
	}
}

// Function for memory allocation and data initialization
void ProcessInitialization(double* &original, double* &pMatrix, double* &pVector, double* &pProcVector,
	double* &pResult, double* &pProcMatrix, double* &pProcResult,
	int &Size, int &RowNum, int &ColumnNum, int p) {
	int RestColumns; // Number of columns, that haven’t been distributed yet
	int i;             // Loop variable

	setvbuf(stdout, 0, _IONBF, 0);
	if (ProcRank == 0) {
		/*do {
			printf("\nEnter size of the initial objects: ");
			scanf("%d", &Size);
			if (Size < ProcNum) {
				printf("Size of the objects must be greater than number of processes! \n ");
			}
		} while (Size < ProcNum);*/
		Size = p;
		int s = determineMaxWidth(ProcNum), q = ProcNum / s;
		RowNum = Size / q, ColumnNum = Size / s;
	}

	MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&RowNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ColumnNum, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Memory allocation
	pProcVector = new double[ColumnNum];
	pProcMatrix = new double[RowNum * ColumnNum];
	pProcResult = new double[RowNum];


	// Obtain the values of initial objects elements
	if (ProcRank == 0) {
		// Inital original matrix
		original = new double[Size * Size];
		// Initial matrix exists only on the pivot process
		pMatrix = new double[Size*Size];
		// Initial vector exists only on the pivot process
		pVector = new double[Size];
		// Result
		pResult = new double[Size];
		// Values of elements are defined only on the pivot process
		RandomDataInitialization(original, pMatrix, pVector, Size, RowNum, ColumnNum);
	}
}

// Function for distribution of the initial objects between the processes
void DataDistribution(double* pMatrix, double* pProcMatrix, double* pVector, double* pProcVector,
	int Size, int RowNum, int ColumnNum) {
	int *pSendNum; // The number of elements sent to the process
	int *pSendInd; // The index of the first data element sent to the process
	int *pSendVecNum; // The number of elements of vector send to the process
	int *pSendVecInd; // The index of the first element in vector sent to the process

	// Alloc memory for temporary objects
	pSendInd = new int[ProcNum];
	pSendNum = new int[ProcNum];
	pSendVecInd = new int[ProcNum];
	pSendVecNum = new int[ProcNum];

	// Define the disposition of the matrix and vector for current process
	pSendNum[0] = RowNum * ColumnNum;
	pSendInd[0] = 0;
	pSendVecNum[0] = ColumnNum;
	pSendVecInd[0] = 0;
	for (int i = 1; i < ProcNum; i++) {

		pSendNum[i] = RowNum * ColumnNum;
		pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
		pSendVecNum[i] = ColumnNum;
		pSendVecInd[i] = (pSendVecInd[i - 1] + pSendVecNum[i - 1]) % Size;
	}

	// Scatter the partial vectors
	MPI_Scatterv(pVector, pSendVecNum, pSendVecInd, MPI_DOUBLE, pProcVector, pSendVecNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Scatter the columns
	MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcMatrix,
		pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Free the memory
	delete[] pSendNum;
	delete[] pSendInd;
	delete[] pSendVecNum;
	delete[] pSendVecInd;
}

// Result vector replication
void ResultReplication(double* pProcResult, double* pResult, int Size, int RowNum, int ColumnNum) {

	int posInRes = (ProcRank / (Size / ColumnNum)) * RowNum;
	double* allProcResult = new double[Size];
	for (int i = 0; i < Size; i++)
		allProcResult[i] = 0;
	for (int i = posInRes; i < posInRes + RowNum; i++) {
		allProcResult[i] = pProcResult[i - posInRes];
	}

	// Sum all the procResult vectors into one pResult vector
	MPI_Reduce(allProcResult, pResult, Size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

// Function for sequential matrix-vector multiplication
void SerialResultCalculation(double* original, double* pVector, double* pResult, int Size) {
	int i, j;  // Loop variables
	for (i = 0; i < Size; i++) {
		pResult[i] = 0;
		for (j = 0; j < Size; j++)
			pResult[i] += original[i * Size + j] * pVector[j];
	}
}

void TestSerialResult() {
	int Size = 5;
	double* original = new double[Size * Size];
	double* pMatrix = new double[Size * Size];
	double* pVector = new double[Size];
	double* pResult = new double[Size];
	int w = determineMaxWidth(Size), h = Size / w;

	RandomDataInitialization(original, pMatrix, pVector, Size, h, w);

	printf("Matrix: \n");
	PrintMatrix(original, Size, Size);
	printf("Vector: \n");
	PrintVector(pVector, Size);

	SerialResultCalculation(original, pVector, pResult, Size);

	PrintVector(pResult, Size);

}

// Process rows and vector multiplication
void ParallelResultCalculation(double* pProcMatrix, double* pProcVector, double* pProcResult, int RowNum, int ColumnNum) {
	int i, j;  // Loop variables
	for (i = 0; i < RowNum; i++)
		pProcResult[i] = 0;
	for (i = 0; i < RowNum; i++) {
		for (j = 0; j < ColumnNum; j++)
			pProcResult[i] += pProcMatrix[i*ColumnNum + j] * pProcVector[j];
	}
}

// Function for formatted matrix output
void PrintMatrix(double* pMatrix, int RowCount, int ColumnCount) {
	int i, j; // Loop variables
	for (i = 0; i < RowCount; i++) {
		for (j = 0; j < ColumnCount; j++)
			printf("%7.4f ", pMatrix[i * RowCount + j]);
		printf("\n");
	}
}

// Function for formatted vector output
void PrintVector(double* pVector, int Size) {
	int i;
	for (i = 0; i < Size; i++)
		printf("%0.6f ", pVector[i]);
	printf("\n");
}

void TestDistribution(double* original, double* pVector, double* pProcVector, double* pProcMatrix,
	int Size, int RowNum, int ColumnNum) {
	if (ProcRank == 0) {
		printf("Initial Matrix: \n");
		PrintMatrix(original, Size, Size);
		printf("Initial Vector: \n");
		PrintVector(pVector, Size);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < ProcNum; i++) {
		if (ProcRank == i) {
			printf("\nProcRank = %d \n", ProcRank);
			printf(" Matrix Stripe:\n");
			PrintMatrix(pProcMatrix, RowNum, ColumnNum);
			printf(" Vector: \n");
			PrintVector(pProcVector, ColumnNum);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

// Fuction for testing the results of multiplication of the matrix stripe 
// by a vector
void TestPartialResults(double* pProcResult, int RowNum) {
	int i;    // Loop variables
	for (i = 0; i < ProcNum; i++) {
		if (ProcRank == i) {
			printf("\nProcRank = %d \n Part of result vector: \n", ProcRank);
			PrintVector(pProcResult, RowNum);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

// Testing the result of parallel matrix-vector multiplication

void TestResult(double* original, double* pVector, double* pResult,
	int Size) {
	// Buffer for storing the result of serial matrix-vector multiplication
	double* pSerialResult;
	// Flag, that shows wheather the vectors are identical or not
	int equal = 0;
	int i;                 // Loop variable

	if (ProcRank == 0) {
		pSerialResult = new double[Size];
		SerialResultCalculation(original, pVector, pSerialResult, Size);
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
void ProcessTermination(double* original, double* pMatrix, double* pVector, double* pProcVector, double* pResult,
	double* pProcMatrix, double* pProcResult) {
	if (ProcRank == 0) {
		delete[] pMatrix;
		delete[] pVector;
		delete[] pResult;
		delete[] original;
	}
	delete[] pProcVector;
	delete[] pProcResult;
	delete[] pProcMatrix;
}

void main(int argc, char* argv[])
{
	double* originalMatrix; // The first argument - initial matrix
	double* pMatrix;  // Tranfsormed matrix
	double* pVector;  // The second argument - initial vector
	double* pProcVector; // The partial vector on the current process
	double* pResult;  // Result vector for matrix-vector multiplication 
	int Size;		    // Sizes of initial matrix and vector
	double* pProcMatrix;   // Stripe of the matrix on the current process
	double* pProcResult; // Block of the result vector on the current process
	int RowNum;          // Number of rows in the matrix block
	int ColumnNum;          // Number of columns in the matrix block
	double Duration, Start, Finish;

	int N[] = { 10,100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 };

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	for (int i = 0; i < 12; i++) {
		if (ProcRank == 0) {
			printf("Parallel matrix-vector multiplication program\n");
		}

		// Memory allocation and data initialization
		ProcessInitialization(originalMatrix, pMatrix, pVector, pProcVector, pResult, pProcMatrix, pProcResult,
			Size, RowNum, ColumnNum, N[i]);

		Start = MPI_Wtime();

		// Distributing the initial objects between the processes
		DataDistribution(pMatrix, pProcMatrix, pVector, pProcVector, Size, RowNum, ColumnNum);

		//TestDistribution(originalMatrix, pVector, pProcVector, pProcMatrix, Size, RowNum, ColumnNum);
		// Process rows and vector multiplication
		ParallelResultCalculation(pProcMatrix, pProcVector, pProcResult, RowNum, ColumnNum);

		// Result replication
		ResultReplication(pProcResult, pResult, Size, RowNum, ColumnNum);

		Finish = MPI_Wtime();
		Duration = Finish - Start;

		//TestPartialResults(pProcResult, RowNum);
		TestResult(originalMatrix, pVector, pResult, Size);

		if (ProcRank == 0) {
			printf("Time of execution = %f\n", Duration);
		}




		//TestSerialResult();
		// Process termination
		ProcessTermination(originalMatrix, pMatrix, pVector, pProcVector, pResult, pProcMatrix, pProcResult);
	}
	MPI_Finalize();
}
