# Multiclass Dry Bean Classification & Deployment System

## Problem Statement
Automated seed classification is essential in modern agricultural processing to replace labor-intensive manual inspection. This project implements six machine learning models to classify high-dimensional computer vision geometric features of dry beans into seven distinct commercial varieties.

## Dataset Description
- **Source:** UCI Machine Learning Repository (Dry Bean Dataset)
- **Instances:** 13,611
- **Features:** 16 morphological attributes (e.g., Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1–4)
- **Target Classes (7):** SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON

## GitHub Repository & Live Demo
- **GitHub Repository:** https://github.com/your-username/dry-bean-classification-streamlit
- **Live Streamlit Application:** https://your-app-name.streamlit.app

## ML Models Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.8917 | 0.9448 | 0.9084 | 0.9088 | 0.9085 | 0.8691 |
| kNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Random Forest (Ensemble) | **0.9218** | 0.9929 | **0.9362** | **0.9323** | **0.9341** | **0.9054** |
| Gradient Boosting (Ensemble) | 0.9207 | **0.9935** | 0.9360 | 0.9309 | 0.9333 | 0.9040 |

## Model Performance Observations

| ML Model Name | Performance Observations |
| :--- | :--- |
| **Logistic Regression** | Performed surprisingly strong (92.14% accuracy) after feature scaling due to strong linear relationships among area and perimeter dimensions. |
| **Decision Tree** | Suffered from slight overfitting on training splits, recording the lowest overall MCC score (0.8691). |
| **kNN** | Achieved competitive 91.66% accuracy; highly dependent on exact $z$-score feature standardization across high-dimensional space. |
| **Naive Bayes** | High AUC (0.9916) but slightly lower accuracy (89.79%) due to feature correlation among shape factors violating the strict conditional independence assumption. |
| **Random Forest (Ensemble)** | **Overall Winner.** Delivered highest Accuracy (0.9218), F1 Score (0.9341), and MCC (0.9054) by effective bagging across decorrelated decision trees. |
| **Gradient Boosting (Ensemble)** | Achieved highest AUC (0.9935) and matched Random Forest closely, showing robust sequential boosting capabilities. |