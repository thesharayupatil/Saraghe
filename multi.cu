#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include "cuda_runtime.h"

// Define matrix size
#define N 16

__global__ void matrix_multiply(float *a, float *b, float *c) {
    // Calculate thread index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the product of two matrices
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}

void matrixMultiplication(float *a ,float *b ,float *c) {
 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += a[i * N+ k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
    
       // Print the result matrix
          printf("\nMatrix result using normal function : \n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", c[i * N + j]);
        }
        printf("\n");
    }
    printf("\n-----------------------------------------------------------------------");
    
}



int main() {
    float *a, *b, *c,*d;  // Pointers to matrices in host memory
    float *dev_a, *dev_b, *dev_c;  // Pointers to matrices in device memory
    int size = N * N * sizeof(float);

    // Allocate memory for matrices in host memory
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);
    d = (float *)malloc(size);

    // Initialize matrices with random values
    for (int i = 0; i < N * N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // Allocate memory for matrices in device memory
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    // Copy matrices from host memory to device memory
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 dimGrid(N / 16, N / 16);
    dim3 dimBlock(16, 16);

    // Call the kernel function

    clock_t tic, toc;
    tic = clock();
    matrix_multiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c);
    toc = clock();

    float timeTakenGPU = ((float)(toc - tic)) / CLOCKS_PER_SEC;

    // Copy the result matrix from device memory to host memory
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

       // Print the A matrix
       printf("Matrix A : \n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n---------------------------------------------------------------------------------\n");

       // Print the B matrix
       printf("Matrix B : \n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", b[i * N + j]);
        }
        printf("\n");
    }
      printf("\n---------------------------------------------------------------------------------");

    // normal 
     // CPU
   
    tic = clock();
    matrixMultiplication(a,b,d);
    toc = clock();

  float timeTakenCPU =(float) ((toc - tic)) / CLOCKS_PER_SEC;
    
    // Print the result matrix parallel
       printf("\nMatrix Result using cuda : \n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", c[i * N + j]);
        }
        printf("\n");
    }
    printf("----------------------------------------------------------------------------------\n");
  
    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    printf("\n ");
   printf("CPU Time: %f \n", timeTakenCPU);
   printf("GPU Time: %f \n", timeTakenGPU);
   printf("Speed Up: %f \n", timeTakenCPU/timeTakenGPU);
    return 0;
}
