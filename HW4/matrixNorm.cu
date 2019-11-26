#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16
#define GRID_SIZE 1

int printMatrix(int *A, int N){
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

int main(int argc, char const *argv[]){
  printf("Begin\n");
  int N = 4; //Matrix size
  
  srand(300); // Fixed Seed
  // allocate memory in host RAM
  int *A, *B;
  A = (int*)malloc(N*N*sizeof(int));
  B = (int*)malloc(N*N*sizeof(int));
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(GRID_SIZE, GRID_SIZE); 
  //generate matrix A
  printf("generating A\n");
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      A[i * N + j] = rand() / 32768;
    }
  }

  printf("A generated\n");
  printMatrix(A, N);
  printMatrix(B, N); 
  float gpu_elapsed_time_ms;

  //events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  //allocate memory on the device
  int *d_a, *d_b;
  cudaMalloc((void **)&d_a, sizeof(int)*N*N);
  cudaMalloc((void **)&d_b, sizeof(int)*N*N);


  cudaMemcpy(d_a, A, sizeof(int)*N*N, cudaMemcpyHostToDevice);
  
  cudaEventRecord(start, 0);
  // Call Matrix Norm function
  int numBlocks = N / 2;
  int numThreads = N / 2;
  gpu_matrixNorm<<<dimBlock, dimGrid>>>(d_a, d_b, N);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  
  cudaMemcpy(B, d_b, sizeof(int)*N*N, cudaMemcpyDeviceToHost);

  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printMatrix(B, N);
  printf("Runtime = %f ms.\n", gpu_elapsed_time_ms);
  
  cudaFree(d_a);
  cudaFree(d_b);
  free(A);
  free(B);
  return 0;
}
