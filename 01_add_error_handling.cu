#include <stdio.h>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__
void doubleElements(int *a, int N)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N + stride; i += stride)
  {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  /*
   * Add error handling to this source code to learn what errors
   * exist, and then correct them. Googling error messages may be
   * of service if actions for resolving them are not clear to you.
   */

  int N = 10000;
  int *a;
  cudaError_t err;

  size_t size = N * sizeof(int);
  err = cudaMallocManaged(&a, size);
  
  if(err != cudaSuccess){
    printf("Error: %s\n", cudaGetErrorString(err));
  }
  
  init(a, N);

  /*
   * Error checking reveals that the thread size was out of range
   * threads are limited to 1024 or less
   *
   */
  
  /* size_t threads_per_block = 2048; */
  size_t threads_per_block = 1024;
  size_t number_of_blocks = 32;
  
  /* 
   * The example code used two error variables, one before sync and one after.
   * There were no sync errors though, just the one above
   */

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  err = cudaGetLastError();
  if(err != cudaSuccess){
    printf("Error: %s\n", cudaGetErrorString(err));
  }
  
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}
