#include<stdio.h>
#include<math.h>

#define WIDTH 32
#define TILE_WIDTH 16

__global__
void tiledMatMulKernel(float* M, float* N, float* P, int W) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float P_val = 0;

    for(int ph = 0; ph < W / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * W + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * W + Col];

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; ++k) {
            P_val += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    P[Row * W + Col] = P_val;
}

void MatMul(float* M_h, float* N_h, float* P_h, int W) {
    float *M_d, *N_d, *P_d;
    int size = W * W * sizeof(float);

    cudaMalloc((void **) &M_d, size);
    cudaMalloc((void **) &N_d, size);
    cudaMalloc((void **) &P_d, size);

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((W + 15) / 16, (W + 15) / 16);

    tiledMatMulKernel<<<gridDim, blockDim>>>(M_d, N_d, P_d, W);

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