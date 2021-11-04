# cudatest
Playing with cuda

Compile 01_hello_gpu.cu with:

!nvcc -arch=sm_70 -o hello-gpu 01-hello/01-hello-gpu.cu -run

Or on my home box:

!nvcc -o hello-gpu 01-hello/01-hello-gpu.cu -run


Compile 01_first_parallel.cu with:

!nvcc -arch=sm_70 -o first-parallel 02-first-parallel/01-first-parallel.cu -run



CUDA Error Handling Function

It can be helpful to create a macro that wraps CUDA function calls for checking errors. Here is an example, feel free to use it in the remaining exercises:

#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{

/*

 * The macro can be wrapped around any function returning
 
 * a value of type `cudaError_t`.
 
 */

  checkCuda( cudaDeviceSynchronize() )
}
