// mantiuk_naive.cu — CUDA version using only global memory (naive)
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define BLOCK 16
#define IDX(x, y, w) ((y) * (w) + (x))

__global__ void computeLogLuminanceKernel(const float3* rgb, float* logLum, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = IDX(x, y, w);
    float3 p = rgb[idx];
    float lum = 0.2126f * p.x + 0.7152f * p.y + 0.0722f * p.z;
    logLum[idx] = logf(1.f + lum);
}

__global__ void computeGradientKernel(const float* logLum, float* gx, float* gy, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w - 1 || y >= h - 1) return;
    int idx = IDX(x, y, w);
    gx[idx] = logLum[idx + 1] - logLum[idx];
    gy[idx] = logLum[idx + w] - logLum[idx];
}

__global__ void compressGradientKernel(float* gx, float* gy, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    gx[idx] = gx[idx] / (1.f + alpha * fabsf(gx[idx]));
    gy[idx] = gy[idx] / (1.f + alpha * fabsf(gy[idx]));
}

__global__ void divergenceKernel(const float* gx, const float* gy, float* div, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= w - 1 || y >= h - 1) return;
    int idx = IDX(x, y, w);
    div[idx] = (gx[idx] - gx[idx - 1]) + (gy[idx] - gy[idx - w]);
}

__global__ void poissonJacobiNaive(float* u, float* uNew, const float* div, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= w - 1 || y >= h - 1) return;
    int idx = IDX(x, y, w);
    float sum4 = u[IDX(x + 1, y, w)] + u[IDX(x - 1, y, w)] +
                 u[IDX(x, y + 1, w)] + u[IDX(x, y - 1, w)];
    uNew[idx] = 0.25f * (sum4 - div[idx]);
}

void runMantiukNaive(const Mat& input, Mat& output, float alpha, float offset, float gammaDen, int iterations) {
    int w = input.cols, h = input.rows, N = w * h;
    float gamma = 1.0f / gammaDen;

    float3* h_rgb = new float3[N];
    for (int i = 0; i < N; ++i) {
        Vec3b px = input.at<Vec3b>(i / w, i % w);
        h_rgb[i] = make_float3(px[2] / 255.f, px[1] / 255.f, px[0] / 255.f);
    }

    float3 *d_rgb; float *d_log, *d_gx, *d_gy, *d_div, *d_u, *d_u2;
    cudaMalloc(&d_rgb, sizeof(float3) * N);
    cudaMalloc(&d_log, sizeof(float) * N);
    cudaMalloc(&d_gx, sizeof(float) * N);
    cudaMalloc(&d_gy, sizeof(float) * N);
    cudaMalloc(&d_div, sizeof(float) * N);
    cudaMalloc(&d_u, sizeof(float) * N);
    cudaMalloc(&d_u2, sizeof(float) * N);

    cudaMemcpy(d_rgb, h_rgb, sizeof(float3) * N, cudaMemcpyHostToDevice);

    dim3 blk(BLOCK, BLOCK);
    dim3 grd((w + BLOCK - 1) / BLOCK, (h + BLOCK - 1) / BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    computeLogLuminanceKernel<<<grd, blk>>>(d_rgb, d_log, w, h);
    computeGradientKernel<<<grd, blk>>>(d_log, d_gx, d_gy, w, h);
    compressGradientKernel<<<(N + 255) / 256, 256>>>(d_gx, d_gy, alpha, N);
    divergenceKernel<<<grd, blk>>>(d_gx, d_gy, d_div, w, h);
    cudaMemcpy(d_u, d_log, sizeof(float) * N, cudaMemcpyDeviceToDevice);

    for (int i = 0; i < iterations; ++i) {
        poissonJacobiNaive<<<grd, blk>>>(d_u, d_u2, d_div, w, h);
        swap(d_u, d_u2);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Naive version done in " << ms / 1000.0f << " seconds.\n";

    float* h_u = new float[N];
    cudaMemcpy(h_u, d_u, sizeof(float) * N, cudaMemcpyDeviceToHost);

    output = Mat(h, w, CV_8UC3);
    const float EPS = 1e-4f;
    for (int i = 0; i < N; ++i) {
        float mapped = expf(h_u[i]) - 1.0f + offset;
        float3 rgb = h_rgb[i];
        float origLum = 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z + EPS;
        float s = mapped / origLum;
        uchar r = min(255.f, powf(rgb.x * s, gamma) * 255.f);
        uchar g = min(255.f, powf(rgb.y * s, gamma) * 255.f);
        uchar b = min(255.f, powf(rgb.z * s, gamma) * 255.f);
        output.at<Vec3b>(i / w, i % w) = Vec3b(b, g, r);
    }

    cudaFree(d_rgb); cudaFree(d_log); cudaFree(d_gx); cudaFree(d_gy);
    cudaFree(d_div); cudaFree(d_u); cudaFree(d_u2);
    delete[] h_rgb; delete[] h_u;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: ./mantiuk_naive input.png output.png [alpha offset gamma iter]\n";
        return 1;
    }

    string inputPath = argv[1];
    string outputPath = argv[2];
    float alpha = (argc > 3) ? atof(argv[3]) : 0.2f;
    float offset = (argc > 4) ? atof(argv[4]) : 0.01f;
    float gammaDen = (argc > 5) ? atof(argv[5]) : 2.2f;
    int iterations = (argc > 6) ? atoi(argv[6]) : 500;

    Mat input = imread(inputPath, IMREAD_COLOR);
    if (input.empty()) {
        cerr << "Failed to load input image." << endl;
        return 1;
    }

    Mat output;
    runMantiukNaive(input, output, alpha, offset, gammaDen, iterations);
    imwrite(outputPath, output);
    return 0;
}

