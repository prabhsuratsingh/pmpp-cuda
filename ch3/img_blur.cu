#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>

using namespace std;
// nvcc -ccbin g++-10 -std=c++14 img_blur.cu -o img_blur `pkg-config --cflags --libs opencv4`

#define BLUR_SIZE 1
#define CHANNELS 3

__global__
void imageBlurKernel(unsigned char* Pin, unsigned char* Pout, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < W && row < H) {
        int r = 0, g = 0, b = 0;
        int pixels = 0;

        for(int bR = -BLUR_SIZE; bR < BLUR_SIZE+1; ++bR) {
            for(int  bC = -BLUR_SIZE; bC < BLUR_SIZE+1; ++bC) {
                int currRow = row + bR;
                int currCol = col + bC;

                if(currRow >= 0 && currRow < H && currCol >= 0 && currCol < W) {
                    int idx = (currRow * W + currCol) * CHANNELS;

                    r += Pin[idx];
                    g += Pin[idx + 1];
                    b += Pin[idx + 2];

                    ++pixels;
                }
            }
        }
        int outIdx = (row * W + col) * CHANNELS;
        
        Pout[outIdx] = (unsigned char) (r / pixels);
        Pout[outIdx + 1] = (unsigned char) (g / pixels);
        Pout[outIdx + 2] = (unsigned char) (b / pixels);
    }
}

void imageBlur(unsigned char* Pin_h, unsigned char* Pout_h, int W, int H) {
    unsigned char *Pin_d, *Pout_d;

    int size = W * H * CHANNELS * sizeof(unsigned char);

    cudaMalloc((void **) &Pin_d, size);
    cudaMalloc((void **) &Pout_d, size);

    cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y);

    imageBlurKernel<<<gridSize, blockSize>>>(Pin_d, Pout_d, W, H);
    cudaDeviceSynchronize();

    cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost);

    cudaFree(Pout_d);
    cudaFree(Pin_d);
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
    unsigned char* h_Pout = new unsigned char[W * H * CHANNELS];

    imageBlur(h_Pin, h_Pout, W, H);

    cv::Mat blurImage(H, W, CV_8UC3, h_Pout);
    cv::cvtColor(blurImage, blurImage, cv::COLOR_RGB2BGR);


    cv::imwrite("blur_output.jpg", blurImage);

    cout << "Done! Saved as blur_output.jpg" << endl;

    delete[] h_Pout;
    return 0;
}