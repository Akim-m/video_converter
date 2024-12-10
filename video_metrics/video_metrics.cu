#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <float.h> // This defines FLT_MAX


#define BLOCK_SIZE 16

// PSNR Kernel
__global__ void calculate_psnr(const float *original, const float *compressed, int size, float *result) {
    __shared__ float mse_shared[BLOCK_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float diff = 0.0f;
    if (idx < size) {
        diff = original[idx] - compressed[idx];
    }

    mse_shared[threadIdx.x] = diff * diff;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x && i + blockIdx.x * blockDim.x < size; ++i) {
            sum += mse_shared[i];
        }
        atomicAdd(result, sum);
    }
}

// SSIM Helper Function
__device__ float clamp(float x, float min_val, float max_val) {
    return fmaxf(fminf(x, max_val), min_val);
}

// SSIM Kernel
__global__ void calculate_ssim(const float *original, const float *compressed, int size, float *result) {
    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;

    __shared__ float ssim_shared[BLOCK_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float ssim = 0.0f;

    if (idx < size) {
        float mu_x = original[idx];
        float mu_y = compressed[idx];
        float sigma_x = mu_x * mu_x;
        float sigma_y = mu_y * mu_y;
        float sigma_xy = mu_x * mu_y;

        float numerator = (2.0f * mu_x * mu_y + C1) * (2.0f * sigma_xy + C2);
        float denominator = (sigma_x + sigma_y + C1) * (sigma_x + sigma_y + C2);

        ssim = max(0.0f, min(1.0f, numerator / clamp(denominator, 1e-6f, FLT_MAX)));
    }

    ssim_shared[threadIdx.x] = ssim;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x && i + blockIdx.x * blockDim.x < size; ++i) {
            sum += ssim_shared[i];
        }
        atomicAdd(result, sum);
    }
}

// Kernel Launchers
extern "C" {
    void launch_psnr(const float *original, const float *compressed, int size, float *result) {
        float *d_original, *d_compressed, *d_result;
        cudaMalloc(&d_original, size * sizeof(float));
        cudaMalloc(&d_compressed, size * sizeof(float));
        cudaMalloc(&d_result, sizeof(float));

        cudaMemcpy(d_original, original, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_compressed, compressed, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, sizeof(float));

        calculate_psnr<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_original, d_compressed, size, d_result);

        // Fetch result
        float mse;
        cudaMemcpy(&mse, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        // Final PSNR calculation on the CPU
        float max_pixel_value = 1.0f; // Assuming normalized pixels in [0, 1]
        *result = 20.0f * log10f(max_pixel_value / sqrtf(mse / size + 1e-8f));

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

        float ssim_sum;
        cudaMemcpy(&ssim_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        // Final SSIM calculation on the CPU
        *result = ssim_sum / size;

        cudaFree(d_original);
        cudaFree(d_compressed);
        cudaFree(d_result);
    }
}
