#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <arm_sve.h>
#include <benchmark/benchmark.h>
#include <math.h>
#include <omp.h>
#include <chrono>

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

void matMul(T *matA, T *matB, T *matC, int M, int N, int K) {
	int i, j, k;

#pragma omp parallel for collapse(2)
	for (i = 0; i < M; i += 4) {
		for (j = 0; j < M; j += 4) {
			svfloat32_t c00 = svdup_f32(0.0f);
			svfloat32_t c01 = svdup_f32(0.0f);
			svfloat32_t c02 = svdup_f32(0.0f);
			svfloat32_t c03 = svdup_f32(0.0f);
			svfloat32_t c10 = svdup_f32(0.0f);
			svfloat32_t c11 = svdup_f32(0.0f);
			svfloat32_t c12 = svdup_f32(0.0f);
			svfloat32_t c13 = svdup_f32(0.0f);
			svfloat32_t c20 = svdup_f32(0.0f);
			svfloat32_t c21 = svdup_f32(0.0f);
			svfloat32_t c22 = svdup_f32(0.0f);
			svfloat32_t c23 = svdup_f32(0.0f);
			svfloat32_t c30 = svdup_f32(0.0f);
			svfloat32_t c31 = svdup_f32(0.0f);
			svfloat32_t c32 = svdup_f32(0.0f);
			svfloat32_t c33 = svdup_f32(0.0f);
			for (k = 0; k < K; k += 8) {
				svfloat32_t b0 = svld1_f32(svptrue_b32(), matB + (j + 0) * K + k);
				svfloat32_t b1 = svld1_f32(svptrue_b32(), matB + (j + 0) * K + k);
				svfloat32_t b2 = svld1_f32(svptrue_b32(), matB + (j + 0) * K + k);
				svfloat32_t b3 = svld1_f32(svptrue_b32(), matB + (j + 0) * K + k);

				svfloat32_t a0 = svld1_f32(svptrue_b32(), matA + (i + 0) * K + k);
				c00 = svtmad_f32(svptrue_b32(), a0, b0, c00);
				c01 = svtmad_f32(svptrue_b32(), a0, b1, c01);
				c02 = svtmad_f32(svptrue_b32(), a0, b2, c02);
				c03 = svtmad_f32(svptrue_b32(), a0, b3, c03);

				svfloat32_t a1 = svld1_f32(svptrue_b32(), matA + (i + 1) * K + k);
				c10 = svtmad_f32(svptrue_b32(), a1, b0, c10);
				c11 = svtmad_f32(svptrue_b32(), a1, b1, c11);
				c12 = svtmad_f32(svptrue_b32(), a1, b2, c12);
				c13 = svtmad_f32(svptrue_b32(), a1, b3, c13);

				svfloat32_t a2 = svld1_f32(svptrue_b32(), matA + (i + 2) * K + k);
				c20 = svtmad_f32(svptrue_b32(), a2, b0, c20);
				c21 = svtmad_f32(svptrue_b32(), a2, b1, c21);
				c22 = svtmad_f32(svptrue_b32(), a2, b2, c22);
				c23 = svtmad_f32(svptrue_b32(), a2, b3, c23);

				svfloat32_t a3 = svld1_f32(svptrue_b32(), matA + (i + 3) * K + k);
				c20 = svtmad_f32(svptrue_b32(), a3, b0, c30);
				c21 = svtmad_f32(svptrue_b32(), a3, b1, c31);
				c22 = svtmad_f32(svptrue_b32(), a3, b2, c32);
				c23 = svtmad_f32(svptrue_b32(), a3, b3, c33);
			}
		// Store the values to matC
		svst1_f32(svptrue_b32(), matC[(i + 0) * N + j + 0], svaddv_f32(svptrue_b32(), c00));
		svst1_f32(svptrue_b32(), matC[(i + 0) * N + j + 1], svaddv_f32(svptrue_b32(), c01));
		svst1_f32(svptrue_b32(), matC[(i + 0) * N + j + 2], svaddv_f32(svptrue_b32(), c02));
		svst1_f32(svptrue_b32(), matC[(i + 0) * N + j + 3], svaddv_f32(svptrue_b32(), c03));

		svst1_f32(svptrue_b32(), matC[(i + 1) * N + j + 0], svaddv_f32(svptrue_b32(), c10));
		svst1_f32(svptrue_b32(), matC[(i + 1) * N + j + 1], svaddv_f32(svptrue_b32(), c11));
		svst1_f32(svptrue_b32(), matC[(i + 1) * N + j + 2], svaddv_f32(svptrue_b32(), c12));
		svst1_f32(svptrue_b32(), matC[(i + 1) * N + j + 3], svaddv_f32(svptrue_b32(), c13));

		svst1_f32(svptrue_b32(), matC[(i + 2) * N + j + 0], svaddv_f32(svptrue_b32(), c20));
		svst1_f32(svptrue_b32(), matC[(i + 2) * N + j + 1], svaddv_f32(svptrue_b32(), c21));
		svst1_f32(svptrue_b32(), matC[(i + 2) * N + j + 2], svaddv_f32(svptrue_b32(), c22));
		svst1_f32(svptrue_b32(), matC[(i + 2) * N + j + 3], svaddv_f32(svptrue_b32(), c23));

		svst1_f32(svptrue_b32(), matC[(i + 3) * N + j + 0], svaddv_f32(svptrue_b32(), c30));
		svst1_f32(svptrue_b32(), matC[(i + 3) * N + j + 1], svaddv_f32(svptrue_b32(), c31));
		svst1_f32(svptrue_b32(), matC[(i + 3) * N + j + 2], svaddv_f32(svptrue_b32(), c32));
		svst1_f32(svptrue_b32(), matC[(i + 3) * N + j + 3], svaddv_f32(svptrue_b32(), c33));
		}
	}
}

int main() {
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
	auto start = std::chrono::high_resolution_clock::now();
	matMul<float>(A, B, C, M, N, K);
	auto end = std::chrono::high_resolution_clock::now();
	//printMat<float>(C, M, N);
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
	return 0;
}
