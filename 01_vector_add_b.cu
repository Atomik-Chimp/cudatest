#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  
  for(int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  //a = (float *)malloc(size);
  //b = (float *)malloc(size);
  //c = (float *)malloc(size);
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);
  
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);
  
  /*
   * Set up the size of our grid
   *
   */
 
  size_t threadsPerBlock = 256;
  size_t numberOfBlocks = (N + (threadsPerBlock - 1))/ threadsPerBlock;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
  cudaDeviceSynchronize();

  checkElementsAre(7, c, N);

  //free(a);
  //free(b);
  //free(c);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  
}
