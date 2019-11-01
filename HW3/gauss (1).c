/* Gaussian elimination without pivoting.
 * Compile with "gcc gauss.c" 
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#include "mpi.h"

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 3) {
    seed = atoi(argv[2]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  gauss();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

  /* Display timing results */
  printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);

  printf("(CPU times are accurate to the nearest %g ms)\n",
	 1.0/(float)CLOCKS_PER_SEC * 1000.0);
  printf("My total CPU time for parent = %g ms.\n",
	 (float)( (cputstop.tms_utime + cputstop.tms_stime) -
		  (cputstart.tms_utime + cputstart.tms_stime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My system CPU time for parent = %g ms.\n",
	 (float)(cputstop.tms_stime - cputstart.tms_stime) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My total CPU time for child processes = %g ms.\n",
	 (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
		  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");
  
  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gaussWRONG() {
  int norm, row, col, numprocs, rank;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;


  /* Gaussian elimination */
  for (norm = 0; norm < N - 1; norm++) {

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //1. if rank != 0 then iterate over the appropriate section of matrix
    if (rank != 0){
      int rowsTouched = 0;
      for (row = norm + rank - 1; row < N; row=row + numprocs - 1) {
	rowsTouched++;
      	multiplier = A[row][norm] / A[norm][norm];
      	for (col = norm; col < N; col++) {
	  A[row][col] -= A[norm][col] * multiplier;
      	}
      	B[row] -= B[norm] * multiplier;
      }
      // move A and B values to buffer
      // send to rank 0
      int dataSize = rowsTouched * (N - norm);
      int rownum   = 0;
      float * data = (float *)malloc(dataSize * sizeof(float));

      for(row = norm + rank - 1; row < N; row=row + numprocs - 1){
	int baseAddr = rownum * (N-norm);
        for (col = norm; col < N; col++) data[baseAddr + (col - norm)] = A[row][col];
	rownum++;
      }
      
      printf("%d Sent %f\n", rank, *data);
      MPI_Send(data, dataSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
      //TODO: Do we free data?
 
    }else {
      // recieve A and B values
      // move to appropriate location
      for (int process = 1; process < numprocs; process++){
	int rowsUsed = (N / (numprocs - 1)) + 1; // TODO: this leads to unused memory and could be an issue with copying
	int dataSize = (N - norm) * rowsUsed;
	float * data = (float *)malloc(dataSize * sizeof(float));
        MPI_Recv(data, dataSize, MPI_FLOAT, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	printf("Master Received %f\n", *data);

	free(data);
      }
    }
    //2. if rank == 0 then receive matrix info

    MPI_Finalize();
  }
  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */

  //MPI_Finalize();

  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}
















void gauss() {
  int norm, row, col, numprocs, rank, Asize;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;

  Asize = N * N;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //float * data = (float *) malloc(sizeof(float) * N);

  /* Gaussian elimination */
  for (norm = 0; norm < N - 1; norm++) {

    // WAIT TO RECEIVE UPDATED A and B VALUES
    // Broadcast
    MPI_Bcast(A, Asize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(B[norm]), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank != 0){
      for (row = norm + rank; row < N; row=row + numprocs - 1) {
      	multiplier = A[row][norm] / A[norm][norm];
      	for (col = norm; col < N; col++) {
	  	//printf("before - A[%d][%d] = %f\n", row, col, A[row][col]);
	  A[row][col] -= A[norm][col] * multiplier; // TODO: optimize by putting results straight in buffer
	  	//printf("after  - A[%d][%d] = %f\n", row, col, A[row][col]);
	  
      	}
      	B[row] -= B[norm] * multiplier; // TODO: send this out

	// SEND UPDATED ROW TO ROOT
	//memcpy(data, A[row], N*sizeof(float)); // TODO: optimize by removing this line

	MPI_Send(&A[row], N, MPI_FLOAT, 0, row, MPI_COMM_WORLD); // TODO: optimize by sending one message not multiple
	MPI_Send(&B[row], 1, MPI_FLOAT, 0, (N+row), MPI_COMM_WORLD);
      }

    }else {
      // RECEIVE A and B VALUES
      // Parse and apply to correct rows
      for(row = norm + 1; row < N; row++){
        MPI_Recv(&A[row], N, MPI_FLOAT, MPI_ANY_SOURCE, row, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	//memcpy(A[row], data, N*sizeof(float)); // TODO: skip the buffer and load data right into A[row]
	MPI_Recv(&B[row], 1, MPI_FLOAT, MPI_ANY_SOURCE, (N + row), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    // SYNCRONIZE!
    if (rank == 0) print_inputs();
    MPI_Barrier(MPI_COMM_WORLD);

  }
  MPI_Finalize();

  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */

  //MPI_Finalize();

  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}

















void gaussOG() {
  int norm, row, col;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;

  printf("Computing Serially.\n");

  /* Gaussian elimination */
  for (norm = 0; norm < N - 1; norm++) {
    for (row = norm + 1; row < N; row++) {
      multiplier = A[row][norm] / A[norm][norm];
      for (col = norm; col < N; col++) {
	A[row][col] -= A[norm][col] * multiplier;
      }
      B[row] -= B[norm] * multiplier;
    }
    print_inputs();
  }
  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */


  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}
