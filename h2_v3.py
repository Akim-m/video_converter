import sys
from PyQt6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QWidget, QLineEdit, QMessageBox, QComboBox,
    QProgressBar, QGroupBox, QGridLayout, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import subprocess
import os
import re
import cv2
import torch

from torch.nn.functional import conv2d
from torch import Tensor


class ConversionWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, command):
        super().__init__()
        self.command = command
        
    def run(self):
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            duration_regex = re.compile(r"Duration: (\d{2}):(\d{2}):(\d{2})")

            time_regex = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})")
            
            duration_seconds = 0
            
            for line in process.stderr:
                duration_match = duration_regex.search(line)
                if duration_match and duration_seconds == 0:
                    h, m, s = map(int, duration_match.groups())
                    duration_seconds = h * 3600 + m * 60 + s
                
                time_match = time_regex.search(line)
                if time_match and duration_seconds > 0:
                    h, m, s = map(int, time_match.groups())
                    current_seconds = h * 3600 + m * 60 + s
                    progress = int((current_seconds / duration_seconds) * 100)
                    self.progress.emit(progress)
            
            process.wait()
            if process.returncode == 0:
                self.finished.emit()
            else:
                self.error.emit(f"FFmpeg process failed with return code {process.returncode}")
                
        except Exception as e:
            self.error.emit(str(e))

class VideoConverter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Video Converter")
        self.setGeometry(100, 100, 600, 500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout()

        file_layout.addWidget(QLabel("Input File:"), 0, 0)
        self.input_path = QLineEdit()
        self.input_path.setReadOnly(True)
        file_layout.addWidget(self.input_path, 0, 1)
        self.input_button = QPushButton("Browse")
        self.input_button.clicked.connect(self.select_input_file)
        file_layout.addWidget(self.input_button, 0, 2)

        file_layout.addWidget(QLabel("Output File:"), 1, 0)
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Specify the output file")
        file_layout.addWidget(self.output_path, 1, 1)
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.select_output_file)
        file_layout.addWidget(self.output_button, 1, 2)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)


        encoding_group = QGroupBox("Encoding Settings")
        encoding_layout = QGridLayout()

        encoding_layout.addWidget(QLabel("Encoder:"), 0, 0)
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems([
            "H.265 (CPU)",
            "H.265 (GPU)",
            "AV1 (libaom)",
            "AV1 (SVT)",
            "AV1 (NVIDIA NVENC)",
            "AV1 (Intel QSV)"
        ])
        self.encoding_combo.currentTextChanged.connect(self.update_quality_presets)
        encoding_layout.addWidget(self.encoding_combo, 0, 1)

        encoding_layout.addWidget(QLabel("Quality Preset:"), 1, 0)
        self.quality_combo = QComboBox()
        encoding_layout.addWidget(self.quality_combo, 1, 1)

        encoding_layout.addWidget(QLabel("Quality Value:"), 2, 0)
        self.quality_value = QSpinBox()
        self.quality_value.setRange(0, 51)
        self.quality_value.setValue(28)
        encoding_layout.addWidget(self.quality_value, 2, 1)

        encoding_group.setLayout(encoding_layout)
        layout.addWidget(encoding_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        self.convert_button = QPushButton("Start Conversion")
        self.convert_button.clicked.connect(self.convert_video)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_conversion)
        
        button_layout.addWidget(self.convert_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.compare_button = QPushButton("Compare Videos")
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self.compare_videos)
        button_layout.addWidget(self.compare_button)

        self.setLayout(layout)
        self.update_quality_presets()
        self.worker = None

    def calculate_psnr_gpu(original, compressed):
        mse = torch.mean((original - compressed) ** 2)
        max_pixel_value = 1.0  # Assuming normalized images/videos
        psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
        return psnr.item()

    def gaussian_kernel(window_size, sigma):
        gauss = torch.tensor([
            (1 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))) *
            torch.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
            for x in range(window_size)
        ], dtype=torch.float32)
        kernel = gauss / gauss.sum()
        return kernel

    def calculate_ssim_gpu(original, compressed, window_size=11, sigma=1.5):
        kernel = gaussian_kernel(window_size, sigma).to(original.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        mu1 = conv2d(original.unsqueeze(0).unsqueeze(0), kernel, padding=window_size // 2)
        mu2 = conv2d(compressed.unsqueeze(0).unsqueeze(0), kernel, padding=window_size // 2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv2d(original.unsqueeze(0).unsqueeze(0) ** 2, kernel, padding=window_size // 2) - mu1_sq
        sigma2_sq = conv2d(compressed.unsqueeze(0).unsqueeze(0) ** 2, kernel, padding=window_size // 2) - mu2_sq
        sigma12 = conv2d(original.unsqueeze(0).unsqueeze(0) * compressed.unsqueeze(0).unsqueeze(0), kernel, padding=window_size // 2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()


    def update_quality_presets(self):
        self.quality_combo.clear()
        if "GPU" in self.encoding_combo.currentText():
            self.quality_combo.addItems(["Fastest", "Fast", "Medium", "Slow", "Quality"])
        else:
            self.quality_combo.addItems(["Ultrafast", "Fast", "Medium", "Slow", "Veryslow"])

    def select_input_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Input File",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.mov *.wmv);;All Files (*.*)"
        )
        if file_path:
            self.input_path.setText(file_path)
            if not self.output_path.text():
                # Auto-generate output filename
                base, ext = os.path.splitext(file_path)
                self.output_path.setText(f"{base}_converted.mp4")

    def select_output_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Select Output File",
            "",
            "MP4 Files (*.mp4);;MKV Files (*.mkv);;All Files (*.*)"
        )
        if file_path:
            self.output_path.setText(file_path)

    def get_preset_value(self):
        preset_map = {
            "Ultrafast": "ultrafast",
            "Fast": "fast",
            "Medium": "medium",
            "Slow": "slow",
            "Veryslow": "veryslow",
            "Fastest": "p1",
            "Quality": "p7"
        }
        return preset_map.get(self.quality_combo.currentText(), "medium")

    def convert_video(self):
        input_file = self.input_path.text()
        output_file = self.output_path.text()
        encoding_type = self.encoding_combo.currentText()
        quality_value = self.quality_value.value()

        if not input_file or not os.path.exists(input_file):
            QMessageBox.warning(self, "Error", "Please select a valid input file.")
            return

        if not output_file:
            QMessageBox.warning(self, "Error", "Please specify an output file.")
            return

        try:
            preset = self.get_preset_value()
            
            if encoding_type == "H.265 (CPU)":
                command = [
                    "ffmpeg", "-i", input_file, "-c:v", "libx265",
                    "-preset", preset, "-crf", str(quality_value),
                    "-c:a", "aac", "-b:a", "192k", output_file
                ]
            elif encoding_type == "H.265 (GPU)":
                command = [
                    "ffmpeg", "-i", input_file, "-c:v", "hevc_nvenc",
                    "-preset", preset, "-cq", str(quality_value),
                    "-c:a", "aac", "-b:a", "192k", output_file
                ]
            elif encoding_type == "AV1 (libaom)":
                command = [
                    "ffmpeg", "-i", input_file, "-c:v", "libaom-av1",
                    "-cpu-used", preset[1] if "p" in preset else "4",
                    "-crf", str(quality_value), "-b:v", "0",
                    "-c:a", "aac", "-b:a", "192k", output_file
                ]
            elif encoding_type == "AV1 (SVT)":
                command = [
                    "ffmpeg", "-i", input_file, "-c:v", "libsvtav1",
                    "-preset", preset[1] if "p" in preset else "4",
                    "-crf", str(quality_value),
                    "-c:a", "aac", "-b:a", "192k", output_file
                ]
            elif encoding_type == "AV1 (NVIDIA NVENC)":
                command = [
                    "ffmpeg", "-i", input_file, "-c:v", "av1_nvenc",
                    "-cq", str(quality_value), output_file
                ]
            elif encoding_type == "AV1 (Intel QSV)":
                command = [
                    "ffmpeg", "-i", input_file, "-c:v", "av1_qsv",
                    "-preset", preset, "-global_quality", str(quality_value),
                    "-c:a", "aac", "-b:a", "192k", output_file
                ]
            else:
                raise ValueError("Unsupported encoding type selected.")

            self.convert_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Converting...")

            


            self.worker = ConversionWorker(command)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.conversion_finished)
            self.worker.error.connect(self.conversion_error)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
            self.reset_ui()

    def compare_videos(self):
        input_file = self.input_path.text()
        output_file = self.output_path.text()
        
        if not input_file or not output_file:
            QMessageBox.warning(self, "Error", "Both original and compressed videos must exist.")
            return

        try:
            # Load video frames
            def load_frames(video_path):
                cap = cv2.VideoCapture(video_path)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.tensor(frame / 255.0, dtype=torch.float32, device='cuda')
                    frames.append(frame)
                cap.release()
                return torch.stack(frames)

            original_frames = load_frames(input_file)
            compressed_frames = load_frames(output_file)

            # Ensure the same number of frames
            min_frames = min(original_frames.shape[0], compressed_frames.shape[0])
            original_frames = original_frames[:min_frames]
            compressed_frames = compressed_frames[:min_frames]

            # Calculate metrics
            psnr = calculate_psnr_gpu(original_frames, compressed_frames)
            ssim = calculate_ssim_gpu(original_frames, compressed_frames)

            # Calculate size reduction
            original_size = os.path.getsize(input_file)
            compressed_size = os.path.getsize(output_file)
            size_reduction = (original_size - compressed_size) / original_size * 100

            # Show results
            QMessageBox.information(
                self,
                "Comparison Results",
                f"File Size Reduction: {size_reduction:.2f}%\n"
                f"PSNR: {psnr:.2f} dB\n"
                f"SSIM: {ssim:.4f}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during comparison:\n{str(e)}")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Converting... {value}%")

    def conversion_finished(self):
        QMessageBox.information(self, "Success", "Video converted successfully!")
        self.reset_ui()

    def conversion_error(self, error_message):
        QMessageBox.critical(self, "Error", f"Conversion failed:\n{error_message}")
        self.reset_ui()

    def cancel_conversion(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.status_label.setText("Conversion cancelled")
            self.reset_ui()

    def conversion_finished(self):
        QMessageBox.information(self, "Success", "Video converted successfully!")
        self.compare_button.setEnabled(True)
        self.reset_ui()



    def reset_ui(self):
        self.convert_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion') 
    window = VideoConverter()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()