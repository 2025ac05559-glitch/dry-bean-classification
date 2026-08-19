# Multiclass Dry Bean Classification & Deployment System

## Problem Statement
Automated seed classification is essential in modern agricultural processing to replace labor-intensive manual inspection. This project implements six machine learning models to classify high-dimensional computer vision geometric features of dry beans into seven distinct commercial varieties.

## Dataset Description
- **Source:** UCI Machine Learning Repository (Dry Bean Dataset)
- **Instances:** 13,611
- **Features:** 16 morphological attributes (e.g., Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1–4)
- **Target Classes (7):** SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON

## GitHub Repository & Live Demo
- **GitHub Repository:** https://github.com/2025ac05559-glitch/dry-bean-classification
- **Live Streamlit Application:** https://dry-bean-classification-sdmtgqh9dimqblkqdls3c9.streamlit.app/

## ML Models Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.8917 | 0.9448 | 0.9089 | 0.9088 | 0.9088 | 0.8690 |
| kNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Random Forest (Ensemble) | 0.9199 | 0.9930 | 0.9345 | 0.9307 | 0.9325 | 0.9032 |
| Gradient Boosting (Ensemble) | **0.9207** | **0.9935** | **0.9360** | **0.9309** | **0.9333** | **0.9040** |

*Note: The assignment requires 5 models (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest). Gradient Boosting is included as an additional ensemble model for comparison.*

## Model Performance Observations

| ML Model Name | Performance Observations |
| :--- | :--- |
| **Logistic Regression** | Performed strongly (92.14% accuracy, highest AUC among linear/simple models) after feature scaling, since bean shape features separate the 7 classes fairly linearly. |
| **Decision Tree** | Recorded the lowest scores across all metrics (88.9% accuracy, MCC 0.8690), consistent with a single tree overfitting the training split and generalizing worse than ensembles. |
| **kNN** | Achieved competitive 91.66% accuracy; performance is closely tied to feature scaling (StandardScaler) since it relies on distance in feature space. |
| **Naive Bayes** | High AUC (0.9916) but the lowest accuracy among non-tree models (89.79%), reflecting the cost of its conditional-independence assumption when shape features are correlated. |
| **Random Forest (Ensemble)** | Strong all-around performer (91.99% accuracy, MCC 0.9032) via bagging across decorrelated trees, though marginally behind Gradient Boosting and Logistic Regression on this dataset. |
| **Gradient Boosting (Ensemble)** | **Overall Winner.** Best balance of Accuracy (0.9207), AUC (0.9935), Precision (0.9360), Recall (0.9309), F1 (0.9333), and MCC (0.9040) via sequential boosting. |
| **Overall Winner for this dataset** | **Gradient Boosting (Ensemble)** — best or joint-best on 5 of 6 metrics, with Logistic Regression as a close runner-up on Accuracy/AUC. |