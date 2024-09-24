# optimized-matmul

- Optimized matrix multiplication on the quad-core ARM Cortex-A72 CPU in Raspberry Pi 4. This project utilizes various parallel programming techniques, including loop unrolling, Message Passing Interface (MPI), and NEON intrinsics provided by the ARMv8 instruction set. The benchmarking results will be updated in the future.
- To run the repo
  ```
  mkdir build
  cd build
  cmake ../
  make
  mpirun -np 4 ./out
  ```
