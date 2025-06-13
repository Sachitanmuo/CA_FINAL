// mantiuk_shared.cu — CUDA version using shared memory
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define BLOCK 16
#define IDX(x, y, w) ((y) * (w) + (x))

__global__ void poissonJacobiShared(float* u, float* uNew, const float* div, int w, int h) {
    __shared__ float tile[BLOCK + 2][BLOCK + 2];

    int gx = blockIdx.x * BLOCK + threadIdx.x;
    int gy = blockIdx.y * BLOCK + threadIdx.y;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    if (gx < w && gy < h)
        tile[ly][lx] = u[IDX(gx, gy, w)];
    if (threadIdx.x == 0 && gx > 0)
        tile[ly][0] = u[IDX(gx - 1, gy, w)];
    if (threadIdx.x == BLOCK - 1 && gx + 1 < w)
        tile[ly][BLOCK + 1] = u[IDX(gx + 1, gy, w)];
    if (threadIdx.y == 0 && gy > 0)
        tile[0][lx] = u[IDX(gx, gy - 1, w)];
    if (threadIdx.y == BLOCK - 1 && gy + 1 < h)
        tile[BLOCK + 1][lx] = u[IDX(gx, gy + 1, w)];

    __syncthreads();

    if (gx <= 0 || gy <= 0 || gx >= w - 1 || gy >= h - 1) return;
    float sum4 = tile[ly][lx + 1] + tile[ly][lx - 1] + tile[ly + 1][lx] + tile[ly - 1][lx];
    uNew[IDX(gx, gy, w)] = 0.25f * (sum4 - div[IDX(gx, gy, w)]);
}

// 其餘 kernel 和主邏輯都與 naive 版本共用，只差在 Jacobi 實作
// 這裡只列出主函式與 Jacobi 部分為差異化版本

void runMantiukShared(const Mat& input, Mat& output, float alpha, float offset, float gammaDen, int iterations) {
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
        poissonJacobiShared<<<grd, blk>>>(d_u, d_u2, d_div, w, h);
        swap(d_u, d_u2);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Shared memory version done in " << ms / 1000.0f << " seconds.\n";

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
        cerr << "Usage: ./mantiuk_shared input.png output.png [alpha offset gamma iter]\n";
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
    runMantiukShared(input, output, alpha, offset, gammaDen, iterations);
    imwrite(outputPath, output);
    return 0;
}
