# ====================================
# IMPORTS
# ====================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

# ====================================
# LOAD + CLEAN DATA
# ====================================

df = pd.read_csv("breast-cancer.csv")

df = df.drop(columns=["Unnamed: 32"], errors="ignore")
df = df.drop(columns=["id"], errors="ignore")

df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})

# ====================================
# EDA VISUALIZATIONS
# ====================================

# Class distribution
sns.countplot(x=df["diagnosis"])
plt.title("Diagnosis Class Distribution (0 = Benign, 1 = Malignant)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# PCA 2D visualization
pca = PCA(2)
pca_result = pca.fit_transform(df.drop("diagnosis", axis=1))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=df["diagnosis"])
plt.title("PCA 2D - Diagnosis Separation")
plt.show()

# Random Forest feature importance (pre-train)
rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(df.drop("diagnosis", axis=1), df["diagnosis"])
feat_imp = pd.Series(rf_temp.feature_importances_, index=df.drop("diagnosis", axis=1).columns)
feat_imp.sort_values(ascending=False).head(10).plot(kind="bar", figsize=(8,5), title="Top 10 Feature Importances")
plt.show()

# ====================================
# TRAIN / TEST SPLIT
# ====================================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================
# SCALING
# ====================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====================================
# MODELS
# ====================================

log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)
y_pred_log = log.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ====================================
# METRICS
# ====================================

def eval_model(name, y_true, y_pred):
    print(f"\n==== {name} ====")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

eval_model("Logistic Regression", y_test, y_pred_log)
eval_model("Random Forest", y_test, y_pred_rf)

print("\nLogistic Regression Report:\n", classification_report(y_test, y_pred_log))
print("\nRandom Forest Report:\n", classification_report(y_test, y_pred_rf))

# ====================================
# MODEL COMPARISON GRAPH (bar chart)
# ====================================

models = ["Logistic Regression", "Random Forest"]

accuracy = [
    accuracy_score(y_test, y_pred_log),
    accuracy_score(y_test, y_pred_rf)
]

precision = [
    precision_score(y_test, y_pred_log),
    precision_score(y_test, y_pred_rf)
]

recall = [
    recall_score(y_test, y_pred_log),
    recall_score(y_test, y_pred_rf)
]

f1 = [
    f1_score(y_test, y_pred_log),
    f1_score(y_test, y_pred_rf)
]

scores = pd.DataFrame({
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}, index=models)

scores.plot(kind="bar", figsize=(8,5), title="Model Comparison (Logistic vs Random Forest)")
plt.ylabel("Score")
plt.ylim(0,1)
plt.show()

# ====================================
# CONFUSION MATRICES
# ====================================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, cmap="Blues", fmt="d")
plt.title("Logistic Regression CM")

plt.subplot(1,2,2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap="Greens", fmt="d")
plt.title("Random Forest CM")

plt.tight_layout()
plt.show()
