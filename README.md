# 🎨 Multi-Aspect Face Extractor for AI Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-informational)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance, parallelized Python script for extracting faces from images. This tool is specifically designed to create rich datasets for training AI Diffusion Models (e.g., for LoRA) by capturing faces in multiple aspects and framings from a single detection.

<br>

![Example banner showing an original photo on the left and multiple resulting cropped faces on the right](https://user-images.githubusercontent.com/10672785/232288005-7d8a2a8a-4c28-4f8d-a7a2-15f5a5e35f83.png)
*This tool turns one source image into a rich dataset of multiple framings.*

## ✨ Core Features

-   **High-Performance Detection:** Utilizes OpenCV's deep learning **YuNet** face detector via an ONNX model for fast and accurate results.
-   **🚀 Backend Acceleration:** Supports multiple OpenCV DNN backends, including **CUDA** and **CUDA_FP16** for massive speed-ups on NVIDIA GPUs.
-   **🖼️ Multi-Aspect Extraction:** The key feature! Instead of one crop per face, it generates numerous variations (from tight close-ups to wider portraits) to create a diverse training dataset automatically.
-   **⚡ Concurrent Processing:** Employs a `ThreadPoolExecutor` to process your entire image library in parallel, maximizing CPU usage.
-   **✅ Built-in Quality Control:** Each generated crop is re-validated to ensure a face is present, eliminating bad crops from your final dataset.
-   **🧩 Extensive Format Support:** Handles over 50 different image file extensions automatically.

---

## ⚙️ Setup & Installation

### 1. Prerequisites

-   Python 3.8+
-   Git

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/Daemon0ps/yunet_lora_face_extraction.git)
cd your-repo-name
```

### 3. Download the YuNet Model

This script requires the `face_detection_yunet_2023mar.onnx` model file.

-   **Create the required directory path:**
    ```bash
    mkdir -p ./opencv_zoo/models/face_detection_yunet/
    ```
-   **Download the model file** from the [OpenCV Zoo GitHub repository](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). You need to download the file named `face_detection_yunet_2023mar.onnx`.
-   **Place the downloaded `.onnx` file** inside the `face_detection_yunet` directory you just created.

Your final folder structure must look like this:

```
your-repo-name/
├── your_script_name.py
└── opencv_zoo/
    └── models/
        └── face_detection_yunet/
            └── face_detection_yunet_2023mar.onnx
```

### 4. Install Dependencies

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install the required packages
pip install opencv-python numpy Pillow tqdm
```

---

## 🔧 How to Use

Configuration is done by editing the variables at the bottom of the script. No command-line arguments are needed.

### 1. Configure Your Paths & Settings

Open the script and modify the variables in the main execution block (`if __name__ == "__main__":`).

```python
# file_path=_rbs(fr'E:/LORA/dae/train/pics/')
file_path = _rbs(rf"E:/pic_prog/pics/") # <-- SET YOUR INPUT FOLDER HERE

# save_path=_rbs(fr'E:/LORA/dae/train/faces/')
save_path = _rbs(rf"E:/pic_prog/faces/") # <-- SET YOUR OUTPUT FOLDER HERE

# Define the output resolution for the final square images
width_size, height_size = (640, 640)

# Define the output image format
image_type = "jpg" # Can be "png", "webp", etc.
```

### 2. (Optional) Configure Performance Backend

For advanced users with compatible hardware (like NVIDIA GPUs), you can change the processing backend for better performance. The default is `CPU` (Index `0`).

Find this section in the `if __name__ == "__main__":` block:

```python
# --- PERFORMANCE CONFIGURATION ---
BACKEND_TARGET_PAIRS = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],         # 0: Default CPU
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],          # 1: For NVIDIA CUDA
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],     # 2: For NVIDIA CUDA (FP16, faster, less precise)
    # ... other backends
]

# Change the index [0] to select a different pair, e.g., [2] for CUDA FP16
backend_id, target_id = (BACKEND_TARGET_PAIRS[0][0], BACKEND_TARGET_PAIRS[0][1])
```
*Note: Using CUDA backends requires an appropriate NVIDIA driver and CUDA Toolkit version compatible with your OpenCV build.*

### 3. Run the Script

Once configured, execute the script from your terminal:

```bash
python your_script_name.py
```

A progress bar will appear, showing the status of the face extraction process.

---

## 💡 How It Works

The script follows an intelligent pipeline to ensure a high-quality, varied dataset.

1.  **Pad Image:** The source image is first padded with a black border. This allows the cropping algorithm to create wide-angle shots that extend beyond the original image boundaries without errors.
2.  **Detect Face:** The powerful YuNet model detects the primary face(s) in the padded image.
3.  **Generate Aspect Crops:** For each detected face, the script programmatically generates **up to 9 different bounding boxes**, each one slightly larger and with different aspect ratios than the last.
4.  **Validate & Save:** Each of these 9 potential crops is passed through a quality-control check. If a face is still clearly visible, the crop is auto-trimmed to remove black bars, resized to your specified dimensions (e.g., 640x640), and saved to the output directory with a descriptive filename like `original-name_face-000_aspect-01.jpg`.

This "multi-aspect" approach ensures your training data isn't just a series of identical tight headshots, leading to more robust and flexible AI models.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Daemon0ps/yunet_lora_face_extraction/issues).

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
