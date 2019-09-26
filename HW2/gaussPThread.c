#include "gauss.c"
#include <pthread.h>

void elimination(void *argc[]){
  int norm, row, col;
  float mult;

  norm = (int)argc[0];
  row  = (int)argc[1];
  col  = (int)argc[2];
  mult = (float)argc[3];

  A[row][col] -= A[norm][col] * mult;
  
}

void gaussPThread(){
  int norm, row, col;
  float mult;
  int id;

  printf("Computing using PThreads. \n")

  /* Gaussian Elimination*/
  for (norm = 0; norm < N - 1; norm++){
    for (row = norm + 1; row < N; row++){
      mult = A[row][norm] / A[norm][norm];
      pthread_t threads[N-norm];
      col = norm;
      for (id = 0; id < N-norm; id++){
        void * argc[4] = [norm, row, col, mult]
        pthread_create(&threads[id], NULL, elimination, argc);
        col++;
      }
      for (id=0; id< N-norm; id++){
        pthread_join(threads[id], NULL);
      }
      B[row] -= B[norm] * mult;
    }
  }
}
