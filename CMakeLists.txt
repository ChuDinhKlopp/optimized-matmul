cmake_minimum_required(VERSION 3.13)
project(matmul)
set(CMAKE_CXX_FLAGS "-Wall -g -O3 -fopenmp -march=armv8-a+simd")
add_executable(out main.cpp)

find_package(MPI REQUIRED)
target_link_libraries(out PRIVATE MPI::MPI_CXX)
