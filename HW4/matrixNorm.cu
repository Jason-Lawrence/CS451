#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void gpu_matrixNorm(float *A, float *B, int N){
  int row, col;
  float mu, sigma;
  mu = 0.0;
  col = blockIdx.x * blockDim.x + threadIdx.x;
  for(row = 0; row < N; ++row){
    mu += A[row * N + col]
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
  int N = 8; //Matrix size
  
  srand(300); // Fixed Seed
  // allocate memory in host RAM
  int *h_a, *h_b;
  cudaMallocHost((void **) &h_a, sizeof(int)*N*N);
  cudaMallocHost((void **) &h_b, sizeof(int)*N*N);

  //generate matrix A
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      h_a[i * n + j] = rand() / 32768;
    }
  }
  
  float gpu_elapsed_time_ms;

  //events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  //allocate memory on the device
  int *d_a, *d_b;
  cudaMalloc((void **) &d_a, sizeof(int)*N*N);
  cudaMalloc((void **) &d_b, sizeof(int)*N*N);


  cudaMemcpy(d_a, h_a, sizeof(int)*N*N, cudaMemcpyHostToDevice);
  
  cudaEventRecord(start, 0);
  // Call Matrix Norm function
  int numBlocks = N / 4;
  int numThreads = N / 4;
  gpu_matrixNorm<<<numBlocks, numThreads>>>(d_a, d_b, n);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  
  cudaMemcpy(h_b, d_b sizeof(int)*N*N, cudaMemcpyDeviceToHost);

  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("Runtime = %f ms.\n\n", gpu_elapsed_time_ms);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  return 0;
}
