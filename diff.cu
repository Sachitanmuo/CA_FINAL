// mantiuk_cuda.cu with texture support
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define BLOCK 16
#define IDX(x, y, w) ((y) * (w) + (x))

texture<float, 2, cudaReadModeElementType> texU;

__global__ void poissonJacobiTexture(float* uNew, const float* div, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= w - 1 || y >= h - 1) return;

    float sum4 = tex2D(texU, x+1, y) + tex2D(texU, x-1, y) +
                 tex2D(texU, x, y+1) + tex2D(texU, x, y-1);
    uNew[IDX(x, y, w)] = 0.25f * (sum4 - div[IDX(x, y, w)]);
}

void runCudaJacobi(const string& mode, float* d_u, float* d_u2, const float* d_div,
                   int w, int h, int iterations, cudaArray* cuArray = nullptr) {
    dim3 blk(BLOCK, BLOCK);
    dim3 grd((w + BLOCK - 1) / BLOCK, (h + BLOCK - 1) / BLOCK);

    if (mode == "cuda-tex") {
        cudaBindTextureToArray(texU, cuArray);
        for (int i = 0; i < iterations; ++i) {
            poissonJacobiTexture<<<grd, blk>>>(d_u2, d_div, w, h);
            swap(d_u, d_u2);
            cudaMemcpyToArray(cuArray, 0, 0, d_u, sizeof(float) * w * h, cudaMemcpyDeviceToDevice);
        }
        cudaUnbindTexture(texU);
    } else {
        for (int i = 0; i < iterations; ++i) {
            if (mode == "cuda-naive")
                poissonJacobiNaive<<<grd, blk>>>(d_u, d_u2, d_div, w, h);
            else
                poissonJacobiShared<<<grd, blk>>>(d_u, d_u2, d_div, w, h);
            swap(d_u, d_u2);
        }
    }
}

// inside runCudaMantiuk(), insert this before runCudaJacobi:
// --- for cuda-tex ---
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* cuArray = nullptr;
    if (mode == "cuda-tex") {
        cudaMallocArray(&cuArray, &channelDesc, w, h);
        cudaMemcpyToArray(cuArray, 0, 0, d_log, sizeof(float) * N, cudaMemcpyDeviceToDevice);
    }

    runCudaJacobi(mode, d_u, d_u2, d_div, w, h, iterations, cuArray);

    if (cuArray) cudaFreeArray(cuArray);

// end of insert
