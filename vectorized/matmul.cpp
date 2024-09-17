#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <immintrin.h>
#include <benchmark/benchmark.h>
#include <math.h>
#include <omp.h>

template<typename T>
void zeroMat(T *mat, int rows, int cols) {
	for (int i = 0; i < rows * cols; i++) {
		mat[i] = 0;
	}
}

template<typename T>
T hsum(__m256 v) {
	__m128 vhigh, vlow, vresult;
	vlow = _mm256_extractf128_ps(v, 0);
	vhigh = _mm256_extractf128_ps(v, 1);
	vresult = _mm_add_ps(vlow, vhigh);
	// horizontal sum
	vresult = _mm_hadd_ps(vresult, vresult);
	vresult = _mm_hadd_ps(vresult, vresult);
	return _mm_cvtss_f32(vresult);
}

template<typename T>
void vectorizedMatMul(T *matA, T *matB, T *matC, int M, int N, int K) {
	//zeroMat<T>(matC, M, N);
	int i, j, k;

	// unroll outer loops
#pragma omp parallel for collapse(2)
	for (i = 0; i < M; i += 4) {
		for (j = 0; j < N; j += 4) {
			__m256 c00 = _mm256_setzero_ps();
			__m256 c01 = _mm256_setzero_ps();
			__m256 c02 = _mm256_setzero_ps();
			__m256 c03 = _mm256_setzero_ps();
			__m256 c10 = _mm256_setzero_ps();
			__m256 c11 = _mm256_setzero_ps();
			__m256 c12 = _mm256_setzero_ps();
			__m256 c13 = _mm256_setzero_ps();
			__m256 c20 = _mm256_setzero_ps();
			__m256 c21 = _mm256_setzero_ps();
			__m256 c22 = _mm256_setzero_ps();
			__m256 c23 = _mm256_setzero_ps();
			__m256 c30 = _mm256_setzero_ps();
			__m256 c31 = _mm256_setzero_ps();
			__m256 c32 = _mm256_setzero_ps();
			__m256 c33 = _mm256_setzero_ps();
			for (k = 0; k < K; k += 8) {
				__m256 b0 = _mm256_loadu_ps(matB + (j + 0) * K + k);
				__m256 b1 = _mm256_loadu_ps(matB + (j + 1) * K + k);
				__m256 b2 = _mm256_loadu_ps(matB + (j + 2) * K + k);
				__m256 b3 = _mm256_loadu_ps(matB + (j + 3) * K + k);

				__m256 a0 = _mm256_loadu_ps(matA + (i + 0) * K + k);
				c00 = _mm256_fmadd_ps(a0, b0, c00);
				c01 = _mm256_fmadd_ps(a0, b1, c01);
				c02 = _mm256_fmadd_ps(a0, b2, c02);
				c03 = _mm256_fmadd_ps(a0, b3, c03);

				__m256 a1 = _mm256_loadu_ps(matA + (i + 1) * K + k);
				c10 = _mm256_fmadd_ps(a1, b0, c10);
				c11 = _mm256_fmadd_ps(a1, b1, c11);
				c12 = _mm256_fmadd_ps(a1, b2, c12);
				c13 = _mm256_fmadd_ps(a1, b3, c13);

				__m256 a2 = _mm256_loadu_ps(matA + (i + 2) * K + k);
				c20 = _mm256_fmadd_ps(a2, b0, c20);
				c21 = _mm256_fmadd_ps(a2, b1, c21);
				c22 = _mm256_fmadd_ps(a2, b2, c22);
				c23 = _mm256_fmadd_ps(a2, b3, c23);

				__m256 a3 = _mm256_loadu_ps(matA + (i + 3) * K + k);
				c30 = _mm256_fmadd_ps(a3, b0, c30);
				c31 = _mm256_fmadd_ps(a3, b1, c31);
				c32 = _mm256_fmadd_ps(a3, b2, c32);
				c33 = _mm256_fmadd_ps(a3, b3, c33);

			}
			matC[(i + 0) * N + j + 0] = hsum<T>(c00);
			matC[(i + 0) * N + j + 1] = hsum<T>(c01);
			matC[(i + 0) * N + j + 2] = hsum<T>(c02);
			matC[(i + 0) * N + j + 3] = hsum<T>(c03);

			matC[(i + 1) * N + j + 0] = hsum<T>(c10);
			matC[(i + 1) * N + j + 1] = hsum<T>(c11);
			matC[(i + 1) * N + j + 2] = hsum<T>(c12);
			matC[(i + 1) * N + j + 3] = hsum<T>(c13);
			
			matC[(i + 2) * N + j + 0] = hsum<T>(c20);
			matC[(i + 2) * N + j + 1] = hsum<T>(c21);
			matC[(i + 2) * N + j + 2] = hsum<T>(c22);
			matC[(i + 2) * N + j + 3] = hsum<T>(c23);

			matC[(i + 3) * N + j + 0] = hsum<T>(c30);
			matC[(i + 3) * N + j + 1] = hsum<T>(c31);
			matC[(i + 3) * N + j + 2] = hsum<T>(c32);
			matC[(i + 3) * N + j + 3] = hsum<T>(c33);
		}
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

template<class T>
void initMat(T *mat, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			mat[i * cols + j] = rand() % 10;
		}
	}
}

template<typename T>
void transposeMat(T **mat, int rows, int cols) {
	T *tmp = (T *)malloc(cols * rows * sizeof(T));
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			tmp[i * rows + j] = (*mat)[j * cols + i];
		}
	}
	free(*mat);
	*mat = tmp;
}

static void vectorizedMatMul_bench(benchmark::State &s) {
	int M = 1024, N = 1024, K = 1024;
	float *A, *B, *C;
	A = (float *)malloc(M * K * sizeof(float));
	B = (float *)malloc(K * N * sizeof(float));
	C = (float *)malloc(M * N * sizeof(float));
	
	initMat<float>(A, M, K);
	//printMat<float>(A, M, K);
	initMat<float>(B, K, N);
	//printMat<float>(B, K, N);
	transposeMat<float>(&B, K, N);
	//printMat<float>(B, K, N);
	for (auto _: s) {
		vectorizedMatMul<float>(A, B, C, M, N, K);
	}
	//printMat<float>(C, M, N);
}

BENCHMARK(vectorizedMatMul_bench)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
