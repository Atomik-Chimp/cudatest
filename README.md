# cudatest
Playing with cuda

Compile 01_hello_gpu.cu with:

!nvcc -arch=sm_70 -o hello-gpu 01-hello/01-hello-gpu.cu -run

Or on my home box:

!nvcc -o hello-gpu 01-hello/01-hello-gpu.cu -run


Compile 01_first_parallel.cu with:

!nvcc -arch=sm_70 -o first-parallel 02-first-parallel/01-first-parallel.cu -run

