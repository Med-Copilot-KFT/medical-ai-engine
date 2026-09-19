# Medical AI Engine

A modular medical imaging AI engine for processing medical images, running specialized AI models, performing segmentation and detection, and generating AI-assisted outputs for clinical review.

> **Status:** R&D / Prototype

---

## Current Implementation

The current implementation includes:

- Image upload and metadata extraction
- OCR using Tesseract
- Gemini multimodal image analysis
- Flask REST API
- OpenCV-based image/video processing foundation
- NIfTI medical image processing
- MONAI medical imaging pipeline
- PyTorch-based model inference
- 3D CT preprocessing
- Pretrained spleen segmentation model
- 3D U-Net inference
- Sliding-window inference
- Spleen segmentation
- Ground-truth comparison
- Dice score calculation
- Segmentation visualization

The current fully validated medical-AI workflow is:

```text
3D CT Volume
      ↓
Preprocessing
      ↓
MONAI 3D U-Net
      ↓
Sliding-Window Inference
      ↓
Spleen Segmentation
      ↓
Ground-Truth Comparison
      ↓
Dice Score
      ↓
Visualization
```

---

## Technology Stack

### AI & Medical Imaging

| Technology      | Purpose                           | Status      |
| --------------- | ---------------------------------- | ----------- |
| Python 3.11     | Main programming language         | Implemented |
| PyTorch         | Deep learning and model inference | Implemented |
| MONAI           | Medical imaging AI framework      | Implemented |
| MONAI Model Zoo | Pretrained medical AI models      | Implemented |
| MONAI Label     | Medical image annotation/research | Installed   |
| NiBabel         | NIfTI medical volume processing   | Implemented |
| NumPy           | Numerical and array operations    | Implemented |
| SciPy           | Scientific computing              | Supporting  |

### Image & Video Processing

| Technology    | Purpose                         | Status      |
| ------------- | -------------------------------- | ----------- |
| OpenCV        | Image and video processing      | Implemented |
| Pillow        | Image loading and processing    | Implemented |
| Tesseract OCR | Text extraction from images     | Implemented |
| FFmpeg        | Video processing and conversion | Installed   |

### Generative AI

| Technology       | Purpose                        | Status      |
| ---------------- | ------------------------------- | ----------- |
| Google Gemini    | Multimodal image understanding | Implemented |
| Google GenAI SDK | Gemini API integration         | Implemented |

### Backend

| Technology    | Purpose                         | Status      |
| ------------- | -------------------------------- | ----------- |
| Flask         | REST API backend                | Implemented |
| Requests      | HTTP communication              | Supporting  |
| python-dotenv | Environment variable management | Implemented |

---

## Development Tools

| Tool                 | Purpose                         |
| -------------------- | -------------------------------- |
| Visual Studio Code   | Primary development environment |
| macOS Terminal / Zsh | Command-line development        |
| Python `venv`        | Isolated Python environment     |
| pip                  | Python package management       |
| Git                  | Version control                 |
| GitHub               | Source-code repository          |
| cURL                 | API and dataset testing         |
| Thunder Client       | REST API testing                |

---

## Development Environment

* Operating System: macOS
* Machine: MacBook Air
* RAM: 24 GB
* Python: 3.11
* Shell: Zsh
* IDE: Visual Studio Code
* Inference Device: CPU

---

## Project Structure

```text
medical-ai-engine/
│
├── app.py
├── test_gemini.py
├── requirements.txt
├── .env
├── .gitignore
│
├── image_extraction/
│   ├── __init__.py
│   ├── image_processor.py
│   ├── vision_processor.py
│   ├── uploads/
│   └── results/
│
├── medical_ai/
│   ├── __init__.py
│   ├── modality_classifier.py
│   ├── dicom_processor.py
│   ├── dicom_series_processor.py
│   ├── spleen_inference.py
│   └── visualize_spleen.py
│
├── medical_data/
│   ├── dicom_samples/
│   └── volumes/
│
└── videoframes/
    ├── __init__.py
    ├── video_processor.py
    ├── uploads/
    ├── frames/
    └── results/
```

---

# Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Med-Copilot-KFT/medical-ai-engine.git
cd medical-ai-engine
```

## 2. Create Virtual Environment

```bash
python3.11 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If additional medical-imaging packages are required:

```bash
pip install monai monailabel nibabel psutil
```

## 4. Verify Installation

```bash
python --version
```

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
```

```bash
python -c "import monai; print('MONAI:', monai.__version__)"
```

```bash
python -c "import nibabel; print('NiBabel:', nibabel.__version__)"
```

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

---

# Environment Configuration

Create a local `.env` file:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

The API key is used by the Gemini multimodal component.

**Do not commit API keys or other secrets to GitHub.**

---

# Medical Dataset

The current medical-AI demonstration uses:

**Medical Segmentation Decathlon — Task09 Spleen**

Expected structure:

```text
datasets/
└── Task09_Spleen/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    └── dataset.json
```

The current test case is:

```text
imagesTr/spleen_19.nii.gz
labelsTr/spleen_19.nii.gz
```

---

# MONAI Model

The current inference pipeline uses the pretrained:

**MONAI Spleen CT Segmentation Model**

Download the model:

```bash
python -c "from monai.bundle import download; download(name='spleen_ct_segmentation', bundle_dir='models')"
```

The model is used for 3D spleen segmentation from CT volumes.

Expected model location:

```text
models/
└── spleen_ct_segmentation/
    └── models/
        ├── model.pt
        └── model.ts
```

---

# Running the Application

## Start Flask API

```bash
python app.py
```

The API runs locally at:

```text
http://127.0.0.1:5000
```

---

# Test Image Extraction

The image extraction endpoint is:

```text
POST /api/extract/image
```

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/extract/image \
  -F "image=@/path/to/image.png"
```

This tests:

* Image loading
* Image metadata extraction
* OCR

---

# Run Gemini Vision Analysis

Run:

```bash
python test_gemini.py
```

This tests the Gemini multimodal image-analysis component.

The workflow is:

```text
Image
  ↓
Gemini
  ↓
Multimodal Analysis
  ↓
Textual Output
```

---

# Run Medical AI Inference

The main currently validated medical-AI pipeline is:

```bash
python medical_ai/spleen_inference.py
```

This performs:

```text
CT Loading
      ↓
Preprocessing
      ↓
Model Loading
      ↓
3D Inference
      ↓
Spleen Segmentation
      ↓
Original-Space Reconstruction
      ↓
Ground-Truth Comparison
      ↓
Dice Calculation
```

---

# Visualize the Result

After inference completes:

```bash
python medical_ai/visualize_spleen.py
```

Output:

```text
medical_data/volumes/spleen_prediction/
└── spleen_19_visual_comparison.png
```

The visualization compares:

* **Green:** Ground Truth
* **Red:** AI Prediction

---

# Current Medical AI Pipeline

The current pipeline uses:

### Input

3D CT volume in NIfTI format.

### Preprocessing

* Load CT
* Ensure channel-first format
* Convert orientation to RAS
* Resample voxel spacing
* Normalize CT intensity
* Convert to PyTorch tensor

### Model

Pretrained MONAI Spleen CT Segmentation model using a 3D U-Net.

### Inference

MONAI sliding-window inference:

```text
96 × 96 × 96
```

with:

```text
sw_batch_size = 1
overlap = 0.5
```

### Output

A 3D spleen segmentation mask.

### Validation

The predicted segmentation is compared with the provided ground-truth segmentation using the Dice similarity coefficient.

---

# Validation Result

For the tested CT case:

```text
Predicted spleen voxels:  95,869
Ground-truth voxels:      96,852
Intersection voxels:      94,374

Dice Score:               0.9794
```

The Dice score measures the overlap between the AI prediction and the ground-truth segmentation.

The tested case achieved:

```text
Dice = 0.9794
```

This result applies to the tested case and is not a clinical performance claim.
