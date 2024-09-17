cmake_minimum_required(VERSION 3.13)
project(matmul_perf)
set(CMAKE_CXX_FLAGS "-Wall -mavx2 -mfma -g -O3 -ffast-math -fopenmp")
add_executable(out matmul.cpp)

find_package(benchmark REQUIRED)
target_link_libraries(out benchmark::benchmark)
