// mantiuk_hdr.cu  ── CUDA 加速版 Mantiuk Tone Mapping
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"

// ==== 預設參數（可被命令列覆寫）====
#define DEF_ALPHA      0.2f
#define DEF_OFFSET     0.01f
#define DEF_GAMMA_DEN  2.2f    // gamma = 1 / 2.2
#define DEF_JAC_ITERS  500

#define IDX(x, y, w) ((y)*(w)+(x))
#define BLOCK 32     // 任意 16×16 block，方便貼到 shared memory

// ---------- Kernel 區 ---------- //
__global__ void computeLogLuminance(const float3* rgb, float* logLum,
                                    int w, int h)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float3 p = rgb[IDX(x,y,w)];
    float lum = 0.2126f*p.x + 0.7152f*p.y + 0.0722f*p.z;
    logLum[IDX(x,y,w)] = logf(1.f + lum);
}

__global__ void computeGradients(const float* logLum,
                                 float* gx, float* gy,
                                 int w, int h)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= w-1 || y >= h-1) return;

    int id = IDX(x,y,w);
    gx[id] = logLum[id+1]        - logLum[id];
    gy[id] = logLum[id+w]        - logLum[id];
}

__global__ void compressGradients(float* gx, float* gy,
                                  float alpha, int N)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= N) return;

    gx[id] = gx[id] / (1.f + alpha*fabsf(gx[id]));
    gy[id] = gy[id] / (1.f + alpha*fabsf(gy[id]));
}

__global__ void computeDivergence(const float* gx, const float* gy,
                                  float* div, int w, int h)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= w-1 || y >= h-1) return;

    int id = IDX(x,y,w);
    div[id] = (gx[id]            - gx[id-1])
            + (gy[id]            - gy[id-w]);
}

/* ====== Shared-Memory Jacobi ======
 * 每個 block 載入 (BLOCK+2)×(BLOCK+2) 磁貼，四周多 1 像素做邊界。
 */
__global__ void poissonJacobiShared(float* u, float* uNew,
                                    const float* div,
                                    int w, int h)
{
    __shared__ float tile[BLOCK+2][BLOCK+2];

    int gx = blockIdx.x*BLOCK + threadIdx.x;   // global x
    int gy = blockIdx.y*BLOCK + threadIdx.y;   // global y
    int lx = threadIdx.x + 1;                  // local x (+1 for halo)
    int ly = threadIdx.y + 1;                  // local y

    // 載入中心像素
    if (gx < w && gy < h)
        tile[ly][lx] = u[IDX(gx,gy,w)];

    // halo 左右
    if (threadIdx.x == 0 && gx > 0)
        tile[ly][0] = u[IDX(gx-1,gy,w)];
    if (threadIdx.x == BLOCK-1 && gx+1 < w)
        tile[ly][BLOCK+1] = u[IDX(gx+1,gy,w)];

    // halo 上下
    if (threadIdx.y == 0 && gy > 0)
        tile[0][lx] = u[IDX(gx,gy-1,w)];
    if (threadIdx.y == BLOCK-1 && gy+1 < h)
        tile[BLOCK+1][lx] = u[IDX(gx,gy+1,w)];

    // halo 角落 4 個
    if (threadIdx.x==0 && threadIdx.y==0 && gx>0 && gy>0)
        tile[0][0] = u[IDX(gx-1,gy-1,w)];
    if (threadIdx.x==BLOCK-1 && threadIdx.y==0 && gx+1<w && gy>0)
        tile[0][BLOCK+1] = u[IDX(gx+1,gy-1,w)];
    if (threadIdx.x==0 && threadIdx.y==BLOCK-1 && gx>0 && gy+1<h)
        tile[BLOCK+1][0] = u[IDX(gx-1,gy+1,w)];
    if (threadIdx.x==BLOCK-1 && threadIdx.y==BLOCK-1
        && gx+1<w && gy+1<h)
        tile[BLOCK+1][BLOCK+1] = u[IDX(gx+1,gy+1,w)];

    __syncthreads();

    // 避免邊框
    if (gx <= 0 || gy <= 0 || gx >= w-1 || gy >= h-1) return;

    float sum4 = tile[ly][lx+1] + tile[ly][lx-1]
               + tile[ly+1][lx] + tile[ly-1][lx];
    uNew[IDX(gx,gy,w)] = 0.25f*(sum4 - div[IDX(gx,gy,w)]);
}

// ---------- 主程式 ---------- //
int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: ./mantiuk_hdr "
                     "<input.hdr> <output.png> "
                     "[alpha offset gammaDen jacIters]\n";
        return 1;
    }
    float  alpha = (argc > 3) ? std::stof(argv[3]) : DEF_ALPHA;
    float  offset = (argc > 4) ? std::stof(argv[4]) : DEF_OFFSET;
    float  gammaDen = (argc > 5) ? std::stof(argv[5]) : DEF_GAMMA_DEN;
    int    jacIters = (argc > 6) ? std::stoi(argv[6]) : DEF_JAC_ITERS;
    float  gamma = 1.f / gammaDen;

    int w,h,c;  float* in = stbi_loadf(argv[1], &w,&h,&c, 3);
    if (!in) { std::cerr<<"Load error\n"; return 1; }

    int N = w*h;
    float3* h_rgb   = new float3[N];
    for (int i=0;i<N;++i)
        h_rgb[i]={in[i*3],in[i*3+1],in[i*3+2]};

    // ----- CUDA malloc -----
    float3 *d_rgb; float *d_log, *d_gx,*d_gy,*d_div,*d_u,*d_u2;
    cudaMalloc(&d_rgb , sizeof(float3)*N);
    cudaMalloc(&d_log , sizeof(float)*N);
    cudaMalloc(&d_gx  , sizeof(float)*N);
    cudaMalloc(&d_gy  , sizeof(float)*N);
    cudaMalloc(&d_div , sizeof(float)*N);
    cudaMalloc(&d_u   , sizeof(float)*N);
    cudaMalloc(&d_u2  , sizeof(float)*N);

    cudaMemcpy(d_rgb, h_rgb, sizeof(float3)*N, cudaMemcpyHostToDevice);

    dim3 blk(BLOCK,BLOCK);
    dim3 grd((w+BLOCK-1)/BLOCK,(h+BLOCK-1)/BLOCK);

    computeLogLuminance<<<grd,blk>>>(d_rgb,d_log,w,h);
    computeGradients<<<grd,blk>>>(d_log,d_gx,d_gy,w,h);
    compressGradients<<<(N+255)/256,256>>>(d_gx,d_gy,alpha,N);
    computeDivergence<<<grd,blk>>>(d_gx,d_gy,d_div,w,h);

    cudaMemcpy(d_u,d_log,sizeof(float)*N,cudaMemcpyDeviceToDevice);

    for (int i=0;i<jacIters;++i) {
        poissonJacobiShared<<<grd,blk>>>(d_u,d_u2,d_div,w,h);
        std::swap(d_u,d_u2);
    }

    // 回傳結果
    float* h_u = new float[N];
    cudaMemcpy(h_u,d_u,sizeof(float)*N,cudaMemcpyDeviceToHost);

    unsigned char* out = new unsigned char[N*3];
    const float EPS=1e-4f;
    for (int i=0;i<N;++i) {
        float mapped = expf(h_u[i]) - 1.f + offset;
        float3 rgb = h_rgb[i];
        float origLum = 0.2126f*rgb.x + 0.7152f*rgb.y + 0.0722f*rgb.z + EPS;
        float s = mapped / origLum;
        rgb.x = powf(rgb.x * s, gamma);
        rgb.y = powf(rgb.y * s, gamma);
        rgb.z = powf(rgb.z * s, gamma);
        out[i*3+0] = fminf(rgb.x*255.f,255.f);
        out[i*3+1] = fminf(rgb.y*255.f,255.f);
        out[i*3+2] = fminf(rgb.z*255.f,255.f);
    }
    stbi_write_png(argv[2], w,h,3, out, w*3);

    cudaFree(d_rgb);cudaFree(d_log);cudaFree(d_gx);cudaFree(d_gy);
    cudaFree(d_div);cudaFree(d_u);cudaFree(d_u2);
    delete[] in; delete[] h_rgb; delete[] h_u; delete[] out;

    std::cout<<"Done.\n";
    return 0;
}
