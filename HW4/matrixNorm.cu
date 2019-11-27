#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

#define BLOCK_SIZE 16
#define GRID_SIZE 1

int printMatrix(int *A, int N){
  if(N > 8) return 0;
  int i, j;
  printf("\nA = \n\t");
  for(i = 0; i < N; ++i){
    for(j = 0; j < N; ++j){
      if(j < N-1) printf("%d, ", A[i * N + j]);
      else printf("%d;\n\t", A[i * N + j]);
      //printf("%d, ", A[i * N + j], j < N-1 ? ", " : ";\n\t");
    }
  }
  return 0;
}

__global__ void gpu_matrixNorm(int *A, int *B, int N){
  
  int row, col;
  float mu, sigma;
  mu = 0.0;
  col = blockIdx.x * blockDim.x + threadIdx.x;
  if(col < N){
    for(row = 0; row < N; ++row){
      mu += A[row * N + col];
    }
    mu /= (float) N;
    sigma = 0.0;
    for(row = 0; row < N; ++row){
      sigma += powf(A[row * N + col] - mu, 2.0);
    }
    sigma /= (float) N;
    sigma = sqrt(sigma);
    for(row = 0; row < N; ++row){
      if(sigma == 0.0) B[row * N + col] = 0.0;
      else B[row * N + col] = (A[row * N + col] - mu) / sigma;
    }
  }
}

int matrixNorm(){
  int N = 6000; //Matrix size
  
  srand(300); // Fixed Seed

  // allocate memory in host RAM
  int *A, *B;
  A = (int*)malloc(N*N*sizeof(int));
  B = (int*)malloc(N*N*sizeof(int)); 
  //generate matrix A
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      A[i * N + j] = rand() / 32768;
    }
  }
  printMatrix(A, N);
  printMatrix(B, N);
  
  //allocate memory on the device
  int *d_a, *d_b;
  cudaMalloc(&d_a, sizeof(int)*N*N);
  cudaMalloc(&d_b, sizeof(int)*N*N);
  int i, j;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      cudaMemcpy(&d_a[i*N+j], (void*)&A[i*N+j], sizeof(int), cudaMemcpyHostToDevice);
    }
  }
  // Call Matrix Norm function
  int numThreads = 128;
  int numBlocks = ((N*N) + numThreads-1) / numThreads;
  gpu_matrixNorm<<<numBlocks, numThreads>>>(d_a, d_b, N);
  
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      cudaMemcpy((void *)&B[i*N+j], &d_b[i*N+j], sizeof(int), cudaMemcpyDeviceToHost);
    }
  }
  printMatrix(B, N);
  
  cudaFree(d_a);
  cudaFree(d_b);
  free(A);
  free(B);
  return 0;
}

int main(int argc, char const *argv[]){
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  unsigned long long time;
  gettimeofday(&etstart, &tzdummy);

  matrixNorm();

  gettimeofday(&etstop, &tzdummy);
  time = (unsigned long long)(etstop.tv_sec - etstart.tv_sec)*1000000+(etstop.tv_usec - etstart.tv_usec);
  printf("Runtime = %g ms.\n", (float)time/(float)1000);
}
