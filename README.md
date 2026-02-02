# Heart Disease Prediction Using Supervised Machine Learning Algorithms

## 📌 Problem Statement
The objective of this project is to predict whether a patient has heart disease based on various clinical and physiological attributes. This is a supervised machine learning classification problem where the goal is to classify patients into disease and non-disease categories.

---

## 📊 Dataset Description
The dataset used in this project is the **Heart Disease UCI dataset**, obtained from **Kaggle**.  
It consists of **303 records** with **14 attributes** related to patient health.

### 🔹 Features
- Age  
- Sex  
- Chest Pain Type (cp)  
- Resting Blood Pressure (trestbps)  
- Serum Cholesterol (chol)  
- Fasting Blood Sugar (fbs)  
- Resting ECG (restecg)  
- Maximum Heart Rate Achieved (thalach)  
- Exercise Induced Angina (exang)  
- ST Depression (oldpeak)  
- Slope of ST Segment (slope)  
- Number of Major Vessels (ca)  
- Thalassemia (thal)  

### 🎯 Target Variable
- **Target**
  - `0` → No Heart Disease  
  - `1` → Heart Disease Present  

---

## 🧹 Data Cleaning and Preprocessing
The following preprocessing steps were applied before training the machine learning models:

- Checked for missing and invalid values in medically relevant features  
- Replaced invalid zero values in columns such as cholesterol and blood pressure with the median  
- Removed duplicate records to avoid data redundancy  
- Detected outliers using boxplot visualization  
- Applied feature scaling using **StandardScaler**  
- Split the dataset into training and testing sets using an **80:20 ratio**

---

## 🤖 Algorithms Used and Accuracy Results
The following supervised machine learning algorithms were implemented and evaluated:

- **Logistic Regression**  
  - Accuracy: **84.43%**

- **Decision Tree**  
  - Accuracy: **78.69%**

- **Random Forest**  
  - Accuracy: **86.89%**

- **K-Nearest Neighbors (KNN)**  
  - Accuracy: **81.97%**

- **Support Vector Machine (SVM)**  
  - Accuracy: **83.61%**

---

## 📈 Evaluation Metrics and Results
The models were evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-Score  

### 🔍 Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.8443 | 0.85 | 0.83 | 0.84 |
| Decision Tree       | 0.7869 | 0.78 | 0.79 | 0.78 |
| Random Forest       | 0.8689 | 0.87 | 0.86 | 0.86 |
| KNN                 | 0.8197 | 0.82 | 0.81 | 0.81 |
| SVM                 | 0.8361 | 0.84 | 0.83 | 0.83 |

---

## ✅ Conclusion
Among all the implemented models, the **Random Forest classifier** achieved the highest accuracy and F1-score. This indicates that ensemble learning methods are highly effective for heart disease prediction. Proper data preprocessing, handling of invalid values, and feature scaling played a crucial role in improving model performance. The results demonstrate that machine learning techniques can assist in early detection of heart disease, helping healthcare professionals make informed decisions.

