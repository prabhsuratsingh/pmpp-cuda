#include<stdio.h>
#include<math.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < N) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int N) {
    float *A_d, *B_d, *C_d;
    int sz = N * sizeof(float);

    cudaMalloc((void **) &A_d, sz);
    cudaMalloc((void **) &B_d, sz);
    cudaMalloc((void **) &C_d, sz);
    
    cudaMemcpy(A_d, A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sz, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(N/256.0), 256>>>(A_d, B_d, C_d, N);

    cudaMemcpy(C, C_d, sz, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int N = 10; 

    float A[N], B[N], C[N];

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    vecAdd(A, B, C, N);

    printf("Result vector C:\n");
    for (int i = 0; i < N; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    return 0;
}