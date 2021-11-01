#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n");
  } else {
    //printf("Failure. Update the execution configuration as necessary.\n");
    // commented out to prevent failure spam
  }
}

__global__ void printSuccessForCorrectExecutionConfiguration2()
{

  if(threadIdx.x == 1 && blockIdx.x == 1)
  {
    printf("Success!\n");
  } else {
    //printf("Failure. Update the execution configuration as necessary.\n");
    // commented out to prevent failure spam
  }
}

int main()
{
  /*
   * Update the execution configuration so that the kernel
   * will print `"Success!"`.
   * Updated to run enough Kernels
   */

  printSuccessForCorrectExecutionConfiguration<<<256, 1024>>>();
  cudaDeviceSynchronize(); //remember this to let the GPU catch up before the program exits

  /*
   * Do it a second way...
   *
   */
  
  printSuccessForCorrectExecutionConfiguration2<<<2, 2>>>();
  cudaDeviceSynchronize(); //remember this to let the GPU catch up before the program exits 
}
