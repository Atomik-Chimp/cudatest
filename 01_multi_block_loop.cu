#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

__global__ void loop()
{
    printf("This is iteration number %d\n", blockIdx.x*blockDim.x + threadIdx.x);

}

int main()
{
  /*
   * When refactoring `loop` to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this exercise, be sure to use more than 1 block in
   * the execution configuration.
   *
   * Note, sometimes block two finishes first so the sequence is not in order!  
   * Will have to read farther to figure out how to fix these types of problems.
   *
   */

  int N = 10;
  int blocks = 2;
  loop<<<blocks, N/blocks>>>();
  cudaDeviceSynchronize();

}
