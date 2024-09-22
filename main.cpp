#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <arm_neon.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <mpi.h>

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

template<typename T>
void matMul(T *matA, T *matB, T *matC, int M, int N, int K) {
	int i, j, k;

#pragma omp parallel for collapse(2)
	for (i = 0; i < M; i += 4) {
		for (j = 0; j < N; j += 4) {
			float32x4_t c00 = vdupq_n_f32(0.0f);
			float32x4_t c01 = vdupq_n_f32(0.0f);
			float32x4_t c02 = vdupq_n_f32(0.0f);
			float32x4_t c03 = vdupq_n_f32(0.0f);
			float32x4_t c10 = vdupq_n_f32(0.0f);
			float32x4_t c11 = vdupq_n_f32(0.0f);
			float32x4_t c12 = vdupq_n_f32(0.0f);
			float32x4_t c13 = vdupq_n_f32(0.0f);
			float32x4_t c20 = vdupq_n_f32(0.0f);
			float32x4_t c21 = vdupq_n_f32(0.0f);
			float32x4_t c22 = vdupq_n_f32(0.0f);
			float32x4_t c23 = vdupq_n_f32(0.0f);
			float32x4_t c30 = vdupq_n_f32(0.0f);
			float32x4_t c31 = vdupq_n_f32(0.0f);
			float32x4_t c32 = vdupq_n_f32(0.0f);
			float32x4_t c33 = vdupq_n_f32(0.0f);
			for (k = 0; k < K; k += 4) {
				float32x4_t b0 = vld1q_f32(matB + (j + 0) * K + k);
				float32x4_t b1 = vld1q_f32(matB + (j + 0) * K + k);
				float32x4_t b2 = vld1q_f32(matB + (j + 0) * K + k);
				float32x4_t b3 = vld1q_f32(matB + (j + 0) * K + k);

				float32x4_t a0 = vld1q_f32(matA + (i + 0) * K + k);
				c00 = vfmaq_f32(a0, b0, c00);
				c01 = vfmaq_f32(a0, b1, c01);
				c02 = vfmaq_f32(a0, b2, c02);
				c03 = vfmaq_f32(a0, b3, c03);

				float32x4_t a1 = vld1q_f32(matA + (i + 1) * K + k);
				c10 = vfmaq_f32(a1, b0, c10);
				c11 = vfmaq_f32(a1, b1, c11);
				c12 = vfmaq_f32(a1, b2, c12);
				c13 = vfmaq_f32(a1, b3, c13);

				float32x4_t a2 = vld1q_f32(matA + (i + 2) * K + k);
				c20 = vfmaq_f32(a2, b0, c20);
				c21 = vfmaq_f32(a2, b1, c21);
				c22 = vfmaq_f32(a2, b2, c22);
				c23 = vfmaq_f32(a2, b3, c23);

				float32x4_t a3 = vld1q_f32(matA + (i + 3) * K + k);
				c20 = vfmaq_f32(a3, b0, c30);
				c21 = vfmaq_f32(a3, b1, c31);
				c22 = vfmaq_f32(a3, b2, c32);
				c23 = vfmaq_f32(a3, b3, c33);
			}
		// Store the values to matC
		matC[(i + 0) * N + j + 0] = vaddvq_f32(c00);
		matC[(i + 0) * N + j + 1] = vaddvq_f32(c01);
		matC[(i + 0) * N + j + 2] = vaddvq_f32(c02);
		matC[(i + 0) * N + j + 3] = vaddvq_f32(c03);

		matC[(i + 1) * N + j + 0] = vaddvq_f32(c10);
		matC[(i + 1) * N + j + 1] = vaddvq_f32(c11);
		matC[(i + 1) * N + j + 2] = vaddvq_f32(c12);
		matC[(i + 1) * N + j + 3] = vaddvq_f32(c13);

		matC[(i + 2) * N + j + 0] = vaddvq_f32(c20);
		matC[(i + 2) * N + j + 1] = vaddvq_f32(c21);
		matC[(i + 2) * N + j + 2] = vaddvq_f32(c22);
		matC[(i + 2) * N + j + 3] = vaddvq_f32(c23);

		matC[(i + 3) * N + j + 0] = vaddvq_f32(c30);
		matC[(i + 3) * N + j + 1] = vaddvq_f32(c31);
		matC[(i + 3) * N + j + 2] = vaddvq_f32(c32);
		matC[(i + 3) * N + j + 3] = vaddvq_f32(c33);
		}
	}
}

int main() {
	MPI_Init(NULL, NULL);

	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int M = 1024, N = 1024, K = 1024;
	float *A, *B, *C;
	A = (float *)malloc(M * K * sizeof(float));
	B = (float *)malloc(K * N * sizeof(float));
	C = (float *)malloc(M * N * sizeof(float));
	
	int elements_per_proc = M * K / size;
	float *recv_buffer = (float *)malloc(elements_per_proc * sizeof(float));
	float *C_frag = (float *)malloc(elements_per_proc * sizeof(float));

	if (rank == 0) {
		initMat<float>(A, M, K);
		//printMat<float>(A, M, K);
		initMat<float>(B, K, N);
		//printMat<float>(B, K, N);
		transposeMat<float>(&B, K, N);
		//printMat<float>(B, K, N);
	}
	auto mpi_start = MPI_Wtime()
	auto start = std::chrono::high_resolution_clock::now();
	// Scatter matrix A among processes
	MPI_Scatter(A, elements_per_proc, MPI_FLOAT,
			recv_buffer, elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
	// Broadcast matrix B to processes
	MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	matMul<float>(recv_buffer, B, C_frag, M/size, N, K);
	// gather results from processes into matrix C
	MPI_Gather(C_frag, elements_per_proc, MPI_FLOAT,
			C, elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD)
	MPI_Barrier(MPI_COMM_WORLD);
	auto end = std::chrono::high_resolution_clock::now();
	auto mpi_end = MPI_Wtime();
	//printMat<float>(C, M, N);
	MPI_Finalize();
	auto elapsed_time = mpi_end - mpi_start;
    // Only rank 0 prints the timing results
    if (world_rank == 0) {
        printf("Time taken for Parallel matrix multiplication: %f seconds\n", elapsed_time);
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
	}
	return 0;
}
