import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv('dataset/Heart_disease_statlog.csv')

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# ===============================
# DATA PREPROCESSING
# ===============================

# Check for zero values
print("\nZero values count:")
print((df == 0).sum())

# Columns where zero is invalid
cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
df[cols] = df[cols].replace(0, np.nan)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

print("\nMissing values after treatment:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# ===============================
# OUTLIER DETECTION
# ===============================
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['chol'])
plt.title("Outlier Detection for Cholesterol")
plt.show(block=False)
plt.pause(3)
plt.close()

# ===============================
# FEATURE & TARGET SPLIT
# ===============================
X = df.drop('target', axis=1)
y = df['target']

# ===============================
# FEATURE SCALING
# ===============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# TRAIN-TEST SPLIT
# ===============================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# MODEL TRAINING
# ===============================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC()

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "KNN": knn,
    "SVM": svm
}

for model in models.values():
    model.fit(X_train, y_train)

# ===============================
# MODEL EVALUATION
# ===============================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model):
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    )

results = []

for name, model in models.items():
    acc, prec, rec, f1 = evaluate_model(model)
    results.append([name, acc, prec, rec, f1])

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nModel Performance Comparison:")
print(results_df)

input("\nPress Enter to exit...")
