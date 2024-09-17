#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <immintrin.h>
#include <benchmark/benchmark.h>

template<typename T>
void zeroMat(T *mat, int rows, int cols) {
	for (int i = 0; i < rows * cols; i++) {
		mat[i] = 0;
	}
}


template<typename T>
void printMat(T *mat, int rows, int cols) {
	printf("===========\n");
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			std::cout << mat[i * cols + j] << ", ";
		}
		printf("\n");
	}
	printf("===========\n");
}

template<typename T>
void naiveMatMul(T *matA, T *matB, T *matC, int M, int N, int K) {
	zeroMat<T>(matC, M, N);
	for (int i = 0; i < M; i++) {
		for (int k = 0; k < K; k++) {
			for (int j = 0; j < N; j++) {
				matC[i * N + j] += matA[i * K + k] * matB[k * N + j];
			}
		}
	}
}

template<class T>
void initMat(T *mat, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			mat[i * cols + j] = rand() % 10;
		}
	}
}

static void naiveMatMul_bench(benchmark::State &s) {
	// SIMD test 
	int M = 1024, N = 1024, K = 1024;
	int *A, *B, *C;
	A = (int *)malloc(M * K * sizeof(int));
	B = (int *)malloc(K * N * sizeof(int));
	C = (int *)malloc(M * N * sizeof(int));
	
	initMat<int>(A, M, K);
	//printMat<int>(A, M, K);
	initMat<int>(B, K, N);
	//printMat<int>(B, K, N);
	
	for (auto _: s) {
		naiveMatMul<int>(A, B, C, M, N, K);
	}
	//printMat<int>(C, M, N);
}

BENCHMARK(naiveMatMul_bench)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
