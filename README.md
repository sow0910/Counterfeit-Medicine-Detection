# Counterfeit Medicine Detection Using Computer Vision and Siamese Neural Networks

A deep learning-based system for detecting counterfeit medicines by comparing medicine packaging images using a **Siamese Neural Network** with a **ResNet18** backbone. The project includes a **Streamlit web application** for real-time counterfeit medicine verification.

---

## Overview

Counterfeit medicines pose significant risks to public health and can lead to ineffective treatment and severe medical complications. Traditional authentication methods such as barcodes and RFID tags are vulnerable to duplication and require additional infrastructure.

This project proposes an AI-powered solution that analyzes medicine packaging images and determines whether a medicine is **Genuine** or **Counterfeit** using image similarity learning.

---

## Features

- Deep Learning based Counterfeit Detection
- Siamese Neural Network with ResNet18
- Image Similarity using Euclidean Distance
- Real-time Streamlit Web Application
- Detects counterfeit medicines without barcode or RFID
- Confidence score for each prediction
- User-friendly interface

---

## System Architecture

```
Reference Image + Test Image
            │
            ▼
 Image Preprocessing
(Resize + Normalize)
            │
            ▼
 Siamese Neural Network
      (ResNet18)
            │
            ▼
 Feature Embeddings
            │
            ▼
 Euclidean Distance
            │
            ▼
 Threshold (0.7)
            │
     Genuine / Counterfeit
```

---

## Model Architecture

- Siamese Neural Network
- Shared-weight ResNet18 Feature Extractor
- Contrastive Loss
- Euclidean Distance Similarity
- Threshold-based Classification

---

## Dataset

A custom dataset was created manually.

### Genuine Images

- 16 medicine types
- 5–6 images per medicine
- Different lighting conditions
- Multiple viewing angles
- Different distances

### Counterfeit Images

Generated synthetically using:

- Logo modifications
- Text alterations
- Font changes
- Layout modifications
- Blur
- Noise
- Occlusion
- Brightness changes

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| PyTorch | Deep Learning |
| OpenCV | Image Processing |
| Streamlit | Web Application |
| NumPy | Numerical Computing |
| Pandas | Data Processing |
| Matplotlib | Visualization |
| Google Colab | Model Training |

---

## Results

| Metric | Value |
|---------|-------|
| Accuracy | **76.9%** |
| Precision | **81.2%** |
| Recall | **93.3%** |
| F1 Score | **0.86** |

The model achieved high recall, making it suitable for healthcare applications where identifying counterfeit medicines is critical.

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/yourusername/Counterfeit-Medicine-Detection.git
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run app.py
```

---

## Application Workflow

1. Upload Test Medicine Image
2. Upload Genuine Reference Image
3. Images are preprocessed
4. Siamese Network extracts embeddings
5. Euclidean Distance is computed
6. Similarity score is generated
7. Prediction displayed as:

- ✅ Genuine Medicine
- ❌ Counterfeit Medicine

along with a confidence score.

---

## Project Structure

```
Counterfeit_Medicine_Project/
│
├── app.py
├── dashboard.py
├── cv.ipynb
├── siamese_model.pth
├── requirements.txt
├── README.md
├── screenshots/
└── documentation/
```

---

## Limitations

- Limited to 16 medicine types
- Uses synthetically generated counterfeit images
- Performance depends on image quality
- Fixed threshold may require tuning for different datasets

---

## Future Work

- OCR-based text verification
- Mobile application deployment
- Real-world counterfeit dataset
- Triplet Loss implementation
- Blockchain integration for pharmaceutical supply chains
- Barcode/QR verification

---

## Authors

**S. Sowndarya**  
Integrated M.Tech (Computer Science and Engineering - Business Analytics)  
VIT Chennai

**S. Bhavya Sri**  
Integrated M.Tech (Computer Science and Engineering - Business Analytics)  
VIT Chennai

---

## Acknowledgement

Guided by

**Dr. Reena Roy R**

School of Computer Science and Engineering (SCOPE)

VIT Chennai

---

## References

The project is based on research in:

- Siamese Neural Networks
- Computer Vision
- Deep Learning
- ResNet18
- Counterfeit Medicine Detection
- Image Similarity Learning

(Complete references are available in the project documentation.)

---
