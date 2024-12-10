#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <float.h> 

#define BLOCK_SIZE 16

// Gaussian Kernel Generation
__global__ void gaussian_kernel(float *kernel, int window_size, float sigma) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= window_size) return;

    float x = idx - window_size / 2;
    kernel[idx] = expf(-x * x / (2 * sigma * sigma)) / (sigma * sqrtf(2 * M_PI));
}

// PSNR Calculation
__global__ void calculate_psnr(const float *original, const float *compressed, int size, float *result) {
    __shared__ float mse_shared[BLOCK_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float diff = 0;
    if (idx < size) {
        diff = original[idx] - compressed[idx];
    }

    mse_shared[threadIdx.x] = diff * diff;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < blockDim.x && i + blockIdx.x * blockDim.x < size; ++i) {
            sum += mse_shared[i];
        }
        atomicAdd(result, sum);
    }
}

// SSIM Calculation Helper Kernels
__device__ float clamp(float x, float min_val, float max_val) {
    return max(min(x, max_val), min_val);
}

// SSIM Calculation
__global__ void calculate_ssim(const float *original, const float *compressed, int size, float *result) {
    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;

    __shared__ float ssim_shared[BLOCK_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < size) {
        float mu_x = original[idx];
        float mu_y = compressed[idx];
        float sigma_x = mu_x * mu_x;
        float sigma_y = mu_y * mu_y;
        float sigma_xy = mu_x * mu_y;

        float numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2);
        float denominator = (sigma_x + sigma_y + C1) * (sigma_x + sigma_y + C2);

        ssim_shared[threadIdx.x] = numerator / clamp(denominator, 1e-6f, FLT_MAX);
    } else {
        ssim_shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < blockDim.x && i + blockIdx.x * blockDim.x < size; ++i) {
            sum += ssim_shared[i];
        }
        atomicAdd(result, sum);
    }
}

// Kernel launchers
extern "C" {
    void launch_gaussian_kernel(float *kernel, int window_size, float sigma) {
        float *d_kernel;
        cudaMalloc(&d_kernel, window_size * sizeof(float));
        gaussian_kernel<<<(window_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_kernel, window_size, sigma);
        cudaMemcpy(kernel, d_kernel, window_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_kernel);
    }

    void launch_psnr(const float *original, const float *compressed, int size, float *result) {
        float *d_original, *d_compressed, *d_result;
        cudaMalloc(&d_original, size * sizeof(float));
        cudaMalloc(&d_compressed, size * sizeof(float));
        cudaMalloc(&d_result, sizeof(float));

        cudaMemcpy(d_original, original, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_compressed, compressed, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, sizeof(float));

        calculate_psnr<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_original, d_compressed, size, d_result);

        cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_original);
        cudaFree(d_compressed);
        cudaFree(d_result);
    }

    void launch_ssim(const float *original, const float *compressed, int size, float *result) {
        float *d_original, *d_compressed, *d_result;
        cudaMalloc(&d_original, size * sizeof(float));
        cudaMalloc(&d_compressed, size * sizeof(float));
        cudaMalloc(&d_result, sizeof(float));

        cudaMemcpy(d_original, original, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_compressed, compressed, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, sizeof(float));

        calculate_ssim<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_original, d_compressed, size, d_result);

        cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_original);
        cudaFree(d_compressed);
        cudaFree(d_result);
    }
}
