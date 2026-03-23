#include<iostream>
#include<opencv2/opencv.hpp>

#define CHANNELS 3

using namespace std;
// nvcc -ccbin g++-10 -std=c++14 img_greyscale.cu -o img_greyscale `pkg-config --cflags --libs opencv4`

__global__
void colorToGreyscaleKernel(unsigned char* Pout, unsigned char* Pin, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < W && row < H) {
        int greyOffset = row*W + col;

        int rgbOffset = greyOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void colorToGreyscale(unsigned char* h_Pin, unsigned char* h_Pout, int W, int H) {
    unsigned char *d_Pin, *d_Pout;

    size_t colorSize = W * H * CHANNELS * sizeof(unsigned char);
    size_t greySize  = W * H * sizeof(unsigned char);

    cudaMalloc((void**)&d_Pin, colorSize);
    cudaMalloc((void**)&d_Pout, greySize);

    cudaMemcpy(d_Pin, h_Pin, colorSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x,
                  (H + blockSize.y - 1) / blockSize.y);

    colorToGreyscaleKernel<<<gridSize, blockSize>>>(d_Pout, d_Pin, W, H);

    cudaDeviceSynchronize();

    cudaMemcpy(h_Pout, d_Pout, greySize, cudaMemcpyDeviceToHost);

    cudaFree(d_Pin);
    cudaFree(d_Pout);
}

int main() {
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_COLOR);

    if (image.empty()) {
        cout << "Failed to load image!" << endl;
        return -1;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int W = image.cols;
    int H = image.rows;

    unsigned char* h_Pin = image.data;
    unsigned char* h_Pout = new unsigned char[W * H];

    colorToGreyscale(h_Pin, h_Pout, W, H);

    cv::Mat greyImage(H, W, CV_8UC1, h_Pout);

    cv::imwrite("output.jpg", greyImage);

    std::cout << "Done! Saved as output.jpg" << std::endl;

    delete[] h_Pout;
    return 0;
}