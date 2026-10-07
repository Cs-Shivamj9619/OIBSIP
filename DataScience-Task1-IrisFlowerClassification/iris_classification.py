"""
Oasis Infobyte - Data Science Internship
Task 1: Iris Flower Classification
Author: Shivam Umesh Jaiswal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
iris_raw = load_iris()
df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
df['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)

print("--- Dataset Shape ---")
print(df.shape)
print("\n--- Data Types & Non-Null Values ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Missing Value Check ---")
print(df.isnull().sum())

# 2. Exploratory Data Analysis (EDA)
print("\n--- Descriptive Statistics ---")
print(df.describe().T)

print("\n--- Class Distribution ---")
print(df['species'].value_counts())

sns.set_theme(style="whitegrid")
pairplot_fig = sns.pairplot(df, hue='species', diag_kind='kde', palette='tab10')
pairplot_fig.fig.suptitle("Iris Feature Pairplot by Species", y=1.02)
plt.savefig("iris_pairplot.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
for i, feature in enumerate(iris_raw.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=feature, data=df, palette='tab10')
    plt.title(f"{feature} Distribution")
plt.tight_layout()
plt.savefig("iris_boxplots.png", bbox_inches='tight')
plt.close()

# 3. Train/Test Split
X = df[iris_raw.feature_names]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training & Evaluation
# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=200, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)

# Model 2: Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n================ EVALUATION SUMMARY ================")
print(f"Logistic Regression Accuracy: {acc_lr * 100:.2f}%")
print(f"Random Forest Accuracy:       {acc_rf * 100:.2f}%\n")

print("--- Logistic Regression Classification Report ---")
print(classification_report(y_test, y_pred_lr))

print("--- Random Forest Classification Report ---")
print(classification_report(y_test, y_pred_rf))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues',
            xticklabels=iris_raw.target_names, yticklabels=iris_raw.target_names, ax=axes[0])
axes[0].set_title(f"Logistic Regression (Acc: {acc_lr:.2f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens',
            xticklabels=iris_raw.target_names, yticklabels=iris_raw.target_names, ax=axes[1])
axes[1].set_title(f"Random Forest (Acc: {acc_rf:.2f})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("iris_confusion_matrices.png", bbox_inches='tight')
plt.close()
