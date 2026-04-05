#include<stdio.h>
#include<math.h>

#define WIDTH 4

/*Each thread produces one output matrix column*/
__global__
void MatMulKernel(float* M, float* N, float* P, int W) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(col < W) {
        for(int row = 0; row < W; ++row) {
            float P_val = 0;

            for(int k = 0; k < W; ++k) {
                P_val += M[row * W + k] * N[k * W + col];
            }

            P[row * W + col] = P_val;
        }
    }
}

void MatMul(float *M_h, float* N_h, float* P_h, int W) {
    float *M_d, *N_d, *P_d;
    int size = W * W * sizeof(float);

    cudaMalloc((void **) &M_d, size);
    cudaMalloc((void **) &N_d, size);
    cudaMalloc((void **) &P_d, size);

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    dim3 blockDIm(256);
    dim3 gridDim((W + 255)/ 256);

    MatMulKernel<<<gridDim, blockDIm>>>(M_d, N_d, P_d, W);

    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}


int main() {
    int size = WIDTH * WIDTH * sizeof(float);

    float *M_h = (float*)malloc(size);
    float *N_h = (float*)malloc(size);
    float *P_h = (float*)malloc(size);

    printf("Matrix M:\n");
    for(int i = 0; i < WIDTH; i++) {
        for(int j = 0; j < WIDTH; j++) {
            M_h[i * WIDTH + j] = i + j;
            printf("%0.2f ", M_h[i * WIDTH + j]);
        }
        printf("\n");
    }

    printf("\nMatrix N:\n");
    for(int i = 0; i < WIDTH; i++) {
        for(int j = 0; j < WIDTH; j++) {
            N_h[i * WIDTH + j] = i - j;
            printf("%0.2f ", N_h[i * WIDTH + j]);
        }
        printf("\n");
    }

    MatMul(M_h, N_h, P_h, WIDTH);

    printf("\nResult Matrix P = M x N:\n");
    for(int i = 0; i < WIDTH; i++) {
        for(int j = 0; j < WIDTH; j++) {
            printf("%0.2f ", P_h[i * WIDTH + j]);
        }
        printf("\n");
    }

    free(M_h);
    free(N_h);
    free(P_h);

    return 0;
}