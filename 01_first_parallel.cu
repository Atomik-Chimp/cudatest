#include <stdio.h>

/*
 * Refactor firstParallel so that it can run on the GPU.
 * << added __global__ in front of the function >>
 */

__global__ void firstParallel()
{
  printf("This should be running in parallel.\n");
}

int main()
{
  /*
   * Refactor this call to firstParallel to execute in parallel once
   * on the GPU.
   */

  firstParallel<<<1,1>>>();
  cudaDeviceSynchronize();  // is needed or all the white space finishes first
  
  printf("\n"); // add a blank space
  
  /*
   * Refactor this call to firstParallel to execute in parallel in 5 threads
   * on the GPU.
   */
   
  firstParallel<<<1,5>>>();
  cudaDeviceSynchronize();  // is needed or all the white space finishes first

  printf("\n"); // add a blank space
  
  /*
   * Refactor this call to firstParallel to execute in parallel in 5 blocks
   * of 5 threads on the GPU.
   */
  
  firstParallel<<<5,5>>>();
  
  /*
   * Some code is needed below so that the CPU will wait
   * for the GPU kernels to complete before proceeding.
   */
   
   cudaDeviceSynchronize();

}
