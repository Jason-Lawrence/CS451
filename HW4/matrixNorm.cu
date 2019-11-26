#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int printMatrix(int *A, int N){
  int i, j;
  printf("\nA = \n\t");
  for(i = 0; i < N; ++i){
    for(j = 0; j < N; ++j){
      printf("%5.2f%s", A[i * N + j], j < N-1 ? ", " : ";\n\t");
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
  int N = 8; //Matrix size
  
  srand(300); // Fixed Seed
  // allocate memory in host RAM
  int *h_a, *h_b;
  cudaMallocHost((int **) &h_a, sizeof(int)*N*N);
  cudaMallocHost((int **) &h_b, sizeof(int)*N*N);

  //generate matrix A
  printf("generating A\n");
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      h_a[i * N + j] = rand() / 32768;
    }
  }
  printf("A generated\n");
  printMatrix(h_a, N);
  float gpu_elapsed_time_ms;

  //events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  //allocate memory on the device
  int *d_a, *d_b;
  cudaMalloc((int **) &d_a, sizeof(int)*N*N);
  cudaMalloc((int **) &d_b, sizeof(int)*N*N);


  cudaMemcpy(d_a, h_a, sizeof(int)*N*N, cudaMemcpyHostToDevice);
  
  cudaEventRecord(start, 0);
  // Call Matrix Norm function
  int numBlocks = N / 4;
  int numThreads = N / 4;
  gpu_matrixNorm<<<numBlocks, numThreads>>>(d_a, d_b, N);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  
  cudaMemcpy(h_b, d_b, sizeof(int)*N*N, cudaMemcpyDeviceToHost);

  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("Runtime = %f ms.\n\n", gpu_elapsed_time_ms);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  return 0;
}
