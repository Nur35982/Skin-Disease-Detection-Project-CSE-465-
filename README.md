# Skin Disease Detection

## Authors

- **Md. Nurnobi Islam** (ID: 2021803642)
- **Md. Abdullah Al Mahfuz** (ID: 1611869042)  
- **Samsul Islam Niom** (ID: 1620075042)  

**Affiliation:**  
Students, Department of Electrical and Computer Engineering  
North South University, Dhaka, Bangladesh  

---

## Abstract

Skin diseases are critical health concerns requiring accurate and timely diagnosis for effective treatment. This project leverages machine learning and computer vision to develop a skin disease detection system. Using ResNeXt-50 and ResNeXt-101 models trained on the HAM10000 dataset (10,015 skin lesion images), the models achieved **90%** and **88%** accuracy, respectively. These results provide a robust foundation for AI-assisted dermatological diagnosis.

---

## Features

- **Dataset:** HAM10000 (High-quality skin lesion images)  
- **Models:** ResNeXt-50 and ResNeXt-101 (Convolutional Neural Networks with cardinality-based feature aggregation)  
- **Objective:** Support dermatologists with preliminary diagnostic insights  
- **Accuracy:**  
  - ResNeXt-50: **90%** classification accuracy  
  - ResNeXt-101: **88%** classification accuracy  

---

## Model Performance

### 📊 ResNeXt-50 Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| akiec | 0.83 | 0.50 | 0.62 | 30 |
| bcc   | 0.83 | 0.86 | 0.85 | 35 |
| bkl   | 0.80 | 0.69 | 0.74 | 88 |
| df    | 0.83 | 0.62 | 0.71 | 8 |
| nv    | 0.97 | 0.95 | 0.96 | 883 |
| vasc  | 1.00 | 0.69 | 0.82 | 13 |
| mel   | 0.38 | 0.78 | 0.51 | 46 |

- **Accuracy:** 0.90  
- **Macro Avg:** Precision: 0.81 | Recall: 0.73 | F1-score: 0.75  
- **Weighted Avg:** Precision: 0.93 | Recall: 0.90 | F1-score: 0.91  

---

### 📊 ResNeXt-101 Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| akiec | 0.67 | 0.73 | 0.70 | 30 |
| bcc   | 0.85 | 0.83 | 0.84 | 35 |
| bkl   | 0.59 | 0.74 | 0.65 | 88 |
| df    | 0.50 | 0.88 | 0.64 | 8 |
| nv    | 0.99 | 0.92 | 0.95 | 883 |
| vasc  | 1.00 | 0.77 | 0.87 | 13 |
| mel   | 0.38 | 0.65 | 0.48 | 46 |

- **Accuracy:** 0.88  
- **Macro Avg:** Precision: 0.71 | Recall: 0.79 | F1-score: 0.73  
- **Weighted Avg:** Precision: 0.91 | Recall: 0.88 | F1-score: 0.90  

---

## 🔍 Benchmark Comparison

| Method                     | Dataset         | Accuracy |
|----------------------------|------------------|----------|
| Multi-class SVM            | Custom dataset   | 96.25%   |
| CNN                        | ISIC dataset     | 88%      |
| Hybrid AI-Based Localization | N/A            | 97%      |
| MobileNet-v2               | Custom dataset   | 97.5%    |
| **ResNeXt-50** (ours)      | HAM10000         | **90%**  |
| **ResNeXt-101** (ours)     | HAM10000         | **88%**  |

While ResNeXt-50 and ResNeXt-101 perform competitively, there is room for improvement to match top benchmarks.

---

## 🛠 Technologies Used

- **Programming Language:** Python  
- **Framework:** PyTorch  
- **Dataset:** [HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

---

## 🚀 Future Directions

- Optimize ResNeXt architectures or explore deeper variants for higher accuracy  
- Augment HAM10000 dataset with more diverse samples to improve generalization  
- Incorporate patient metadata (e.g., age, gender) for enhanced predictions  
- Experiment with advanced architectures like Transformers or Vision Transformers (ViT)

---

## 📦 Installation and Usage

### ✅ Prerequisites

- Python 3.8 or higher  
- PyTorch  

### 🧪 Steps

1. **Clone the repository**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the HAM10000 dataset** and place it in the project directory.

4. **Train the model**
    ```bash
    python train.py --model resnext50
    # or
    python train.py --model resnext101
    ```

5. **Evaluate the model**
    ```bash
    python evaluate.py --model resnext50
    ```

---

## 📌 Notes

- The HAM10000 dataset is publicly available and can be downloaded from the [ISIC archive](https://www.isic-archive.com/).  
- Ensure a **GPU** is available for faster training with PyTorch.  
- Classification reports reflect performance on a test set of 1103 samples across seven skin lesion classes.

---

## 📫 Contact

For academic or research-related inquiries, please reach out to the authors via North South University.
