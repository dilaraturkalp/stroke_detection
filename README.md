# Stroke Detection from CT Scans

This project aims to detect the presence of stroke (hemorrhagic or ischemic) in brain CT scan images using deep learning techniques.

## Project Overview

- **Dataset Preparation:**  
  The dataset consists of brain CT images categorized as "Stroke Present" (`inme_var`) and "No Stroke" (`inme_yok`). Images are extracted from a zip file, organized, and split into training, validation, and test sets with a 70/15/15 ratio.
- **Model Training:**  
  The model is trained using the [fastai](https://docs.fast.ai/) library on the prepared dataset. The training process involves standard image classification techniques, and the resulting model is exported as `stroke_model.pkl`.
- **Web Application:**  
  The trained model is integrated into a Gradio-based web application. Users can upload a CT scan image, and the app predicts the probability of stroke.

## Usage

1. **Upload** a brain CT scan image through the application interface.
2. **Click** the "Analyze Image" button.
3. **View** the results in the "Diagnosis" and "Results" sections.

## Requirements

- Python 3.8
- fastai <2.8.0
- gradio
- torch
- torchvision
- scikit-image
- numpy==1.21.6
- Pillow
- pathlib

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure the trained model file `stroke_model.pkl` is in the main directory.
2. Run the application:
   ```bash
   python app.py
   ```
3. Open the provided local URL in your browser to use the interface.

## Model Details

- **Framework:** fastai (PyTorch backend)
- **Input:** Brain CT scan images (PNG/JPG)
- **Output:** Probability scores for "Stroke Present" and "No Stroke"
- **Note:** The model expects grayscale or standard RGB CT images.

## Important Note

This tool is for demonstration and educational purposes only. It is **not** intended for actual medical diagnosis. Always consult with a qualified healthcare professional for medical decisions. 