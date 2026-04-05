#include<stdio.h>
#include<math.h>

#define WIDTH 4

__global__
void matMulKernel(float* B, float* C, float* A, int W) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < W)) {
        float P_val = 0;

        for(int k = 0; k < W; ++k) {
            P_val += B[row * W + k] * C[k];
        }

        A[row] = P_val;
    }
}

void MatMul(float* B_h, float* C_h, float* A_h, int W) {
    float *B_d, *C_d, *A_d;
    int size = W * W * sizeof(float);
    int vec_sz = W * sizeof(float);

    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, vec_sz);
    cudaMalloc((void **) &A_d, vec_sz);

    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, vec_sz, cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((W + 255)/ 256);

    matMulKernel<<<gridDim, blockDim>>>(B_d, C_d, A_d, W);

    cudaMemcpy(A_h, A_d, vec_sz, cudaMemcpyDeviceToHost);

    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(A_d);
}


int main() {
    int size = WIDTH * WIDTH * sizeof(float);

    float *B_h = (float*)malloc(size);
    float *C_h = (float*)malloc(WIDTH * sizeof(float));
    float *A_h = (float*)malloc(WIDTH * sizeof(float));

    printf("Matrix B:\n");
    for(int i = 0; i < WIDTH; i++) {
        for(int j = 0; j < WIDTH; j++) {
            B_h[i * WIDTH + j] = i + j;
            printf("%0.2f ", B_h[i * WIDTH + j]);
        }
        printf("\n");
    }

    printf("\nVector C:\n");
    for(int i = 0; i < WIDTH; i++) {
            C_h[i] = i;
            printf("%0.2f ", C_h[i]);
        printf("\n");
    }

    MatMul(B_h, C_h, A_h, WIDTH);

    printf("\nResult Matrix A = B x C:\n");
    for(int i = 0; i < WIDTH; i++) {
            printf("%0.2f ", A_h[i]);
        printf("\n");
    }

    free(B_h);
    free(C_h);
    free(A_h);

    return 0;
}