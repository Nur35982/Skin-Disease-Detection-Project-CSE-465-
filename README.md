Skin Disease Detection
Authors

Md. Nurnobi Islam (ID: 2021803642)
Md. Abdullah Al Mahfuz (ID: 1611869042)  
Samsul Islam Niom (ID: 1620075042)  

AffiliationStudents, Department of Electrical and Computer EngineeringNorth South University, Dhaka, Bangladesh  

Abstract
Skin diseases are critical health concerns requiring accurate and timely diagnosis for effective treatment. This project leverages machine learning and computer vision to develop a skin disease detection system. Using ResNeXt-50 and ResNeXt-101 models trained on the HAM10000 dataset (10,015 skin lesion images), the models achieved 90% and 88% accuracy, respectively. These results provide a robust foundation for AI-assisted dermatological diagnosis.

Features

Dataset: HAM10000 (High-quality skin lesion images).  
Models: ResNeXt-50 and ResNeXt-101 (Convolutional Neural Networks with cardinality-based feature aggregation).  
Objective: Support dermatologists with preliminary diagnostic insights.  
Accuracy: 
ResNeXt-50: 90% classification accuracy.
ResNeXt-101: 88% classification accuracy.




Model Performance
ResNeXt-50 Classification Report
              precision    recall  f1-score   support

       akiec       0.83      0.50      0.62        30
         bcc       0.83      0.86      0.85        35
         bkl       0.80      0.69      0.74        88
          df       0.83      0.62      0.71         8
          nv       0.97      0.95      0.96       883
        vasc       1.00      0.69      0.82        13
         mel       0.38      0.78      0.51        46

    accuracy                           0.90      1103
   macro avg       0.81      0.73      0.75      1103
weighted avg       0.93      0.90      0.91      1103

ResNeXt-101 Classification Report
              precision    recall  f1-score   support

       akiec       0.67      0.73      0.70        30
         bcc       0.85      0.83      0.84        35
         bkl       0.59      0.74      0.65        88
          df       0.50      0.88      0.64         8
          nv       0.99      0.92      0.95       883
        vasc       1.00      0.77      0.87        13
         mel       0.38      0.65      0.48        46

    accuracy                           0.88      1103
   macro avg       0.71      0.79      0.73      1103
weighted avg       0.91      0.88      0.90      1103

Benchmark Comparison
This project compares performance against existing methods:

Multi-class SVM: 96.25% accuracy (custom dataset).  
CNN (ISIC dataset): 88% accuracy.  
Hybrid AI-Based Localization: 97% accuracy.  
MobileNet-v2: 97.5% accuracy (custom dataset).

While ResNeXt-50 (90%) and ResNeXt-101 (88%) perform competitively, there is room for improvement to match top benchmarks.

Technologies Used

Programming Language: Python  
Framework: PyTorch  
Dataset: HAM10000


Future Directions

Optimize ResNeXt architectures or explore deeper variants for higher accuracy.  
Augment the HAM10000 dataset with diverse samples to improve generalization.  
Incorporate patient metadata (e.g., age, gender) for enhanced predictions.  
Experiment with advanced architectures like Transformers or Vision Transformers (ViT).


Installation and Usage
Prerequisites

Python 3.8 or higher  
PyTorch

Steps

Clone the repository:git clone <repository_url>


Install dependencies:pip install -r requirements.txt


Download the HAM10000 dataset and place it in the project directory.
Train the model:python train.py --model resnext50

orpython train.py --model resnext101


Evaluate the model:python evaluate.py --model resnext50




Notes

The HAM10000 dataset is publicly available and can be downloaded from the ISIC archive.
Ensure a GPU is available for faster training with PyTorch.
Classification reports reflect performance on a test set of 1103 samples across seven skin lesion classes.

