import ctypes
import numpy as np

# Load the shared library
image_metrics = ctypes.CDLL('./image_metrics.so')

# Define functions
def gaussian_kernel(window_size, sigma):
    kernel = np.zeros(window_size, dtype=np.float32)
    kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    image_metrics.launch_gaussian_kernel(kernel_ptr, window_size, ctypes.c_float(sigma))
    return kernel

def calculate_psnr(original, compressed):
    size = len(original)
    result = np.zeros(1, dtype=np.float32)
    original_ptr = original.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    compressed_ptr = compressed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    image_metrics.launch_psnr(original_ptr, compressed_ptr, size, result_ptr)
    return result[0]

def calculate_ssim(original, compressed):
    size = len(original)
    result = np.zeros(1, dtype=np.float32)
    original_ptr = original.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    compressed_ptr = compressed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    image_metrics.launch_ssim(original_ptr, compressed_ptr, size, result_ptr)
    return result[0]

# Example usage
if __name__ == "__main__":
    original = np.random.rand(1024).astype(np.float32)
    compressed = np.random.rand(1024).astype(np.float32)

    psnr = calculate_psnr(original, compressed)
    ssim = calculate_ssim(original, compressed)
    kernel = gaussian_kernel(11, 1.5)

    print(f"PSNR: {psnr}")
    print(f"SSIM: {ssim}")
    print(f"Gaussian Kernel: {kernel}")
