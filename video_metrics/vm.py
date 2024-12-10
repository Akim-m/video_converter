import cv2
import ctypes
import numpy as np

# Load the shared library
video_metrics = ctypes.CDLL('./video_metrics.so')

# Define ctypes functions
def calculate_psnr(original, compressed):
    size = len(original)
    result = np.zeros(1, dtype=np.float32)
    original_ptr = original.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    compressed_ptr = compressed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    video_metrics.launch_psnr(original_ptr, compressed_ptr, size, result_ptr)
    return result[0]

def calculate_ssim(original, compressed):
    size = len(original)
    result = np.zeros(1, dtype=np.float32)
    original_ptr = original.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    compressed_ptr = compressed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    video_metrics.launch_ssim(original_ptr, compressed_ptr, size, result_ptr)
    return result[0]

# Process videos in batches
def compare_videos_in_batches(video1, video2, batch_size=16):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    
    psnr_values = []
    ssim_values = []
    frame_count = 0

    while True:
        batch1 = []
        batch2 = []

        for _ in range(batch_size):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Convert to grayscale and normalize
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            batch1.append(frame1.flatten())
            batch2.append(frame2.flatten())

        if not batch1 or not batch2:
            break
        
        # Convert batches to numpy arrays
        batch1 = np.array(batch1, dtype=np.float32)
        batch2 = np.array(batch2, dtype=np.float32)

        # Compute metrics for the batch
        for frame1, frame2 in zip(batch1, batch2):
            psnr = calculate_psnr(frame1, frame2)
            ssim = calculate_ssim(frame1, frame2)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            frame_count += 1
    
    cap1.release()
    cap2.release()
    
    # Calculate average PSNR and SSIM
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    print(f"Processed {frame_count} frames")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

# Run comparison
if __name__ == "__main__":
    compare_videos_in_batches("Mars.mp4", "Mars_converted.mp4")
