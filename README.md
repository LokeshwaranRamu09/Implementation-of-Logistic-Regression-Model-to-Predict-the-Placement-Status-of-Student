# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Collect dataset, split into training and testing sets, and standardize the features.

2.Train the Logistic Regression model using training data.

3.Predict outcomes on test data and compute metrics (accuracy, confusion matrix, ROC–AUC).

4.Validate with cross-validation and use the model for new predictions

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:LOKESHWARAN.R
RegisterNumber:  212224220053


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

np.random.seed(42)
n_samples = 200

cgpa = np.random.uniform(5.0, 10.0, n_samples)
iq = np.random.randint(80, 150, n_samples)
experience = np.random.randint(0, 3, n_samples)
placed = (0.3*cgpa + 0.02*iq + 0.5*experience + np.random.randn(n_samples)) > 7.5
placed = placed.astype(int)

df = pd.DataFrame({
    'cgpa': cgpa,
    'iq': iq,
    'experience': experience,
    'placed': placed
})

print("=== FIRST 10 ROWS OF DATASET ===")
print(df.head(10))

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== DATASET INFO ===")
print(df.info())

print("\n=== DATASET SUMMARY (describe) ===")
print(df.describe())

X = df[['cgpa', 'iq', 'experience']]
y = df['placed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n=== TRAIN/TEST SPLIT ===")
print("Training data size:", X_train.shape[0])
print("Testing data size:", X_test.shape[0])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print("\n=== MODEL COEFFICIENTS ===")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n=== EVALUATION METRICS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

if len(np.unique(y_test)) == 2:
    print("\nROC–AUC Score:", roc_auc_score(y_test, y_prob))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    print("\nFirst 10 ROC Curve Data Points:")
    for i in range(10):
        print(f"Threshold={thresholds[i]:.2f}, FPR={fpr[i]:.2f}, TPR={tpr[i]:.2f}")
else:
    print("\nROC–AUC Score: Cannot be computed (only one class present in test set)")

cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="accuracy")
print("\n=== CROSS-VALIDATION ===")
print("Cross-validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

new_students = [[8.5, 125, 1], [6.2, 95, 0], [7.8, 110, 2], [5.5, 85, 0]]
new_students_scaled = scaler.transform(new_students)
predictions = model.predict(new_students_scaled)

print("\n=== NEW STUDENT PREDICTIONS ===")
for i, pred in enumerate(predictions):
    print(f"Student {i+1} ({new_students[i]}) →", "Placed" if pred == 1 else "Not Placed")

```

## Output:

# FIRST TEN ROES OF DATA SET 
<img width="490" height="258" alt="image" src="https://github.com/user-attachments/assets/cf2bb8f9-36ca-4278-b992-75a6143fddd1" />


# DATASET SHAPE 
<img width="353" height="32" alt="image" src="https://github.com/user-attachments/assets/6e051a87-7260-4577-9b24-2590c1f553e2" />

# DATASET INFO
<img width="745" height="348" alt="image" src="https://github.com/user-attachments/assets/0ea269ce-dcf1-419f-b989-1c0e3738bd68" />

# DATASET SUMMARY (describe)
<img width="865" height="272" alt="image" src="https://github.com/user-attachments/assets/0fa5fb0a-d821-4534-b34a-d23e98668a90" />

# TRAIN/TEST SPLIT
<img width="795" height="82" alt="image" src="https://github.com/user-attachments/assets/afe76215-978e-4db8-8ce4-12dde9d6f168" />

# MODEL COEFFICIENTS
<img width="971" height="91" alt="image" src="https://github.com/user-attachments/assets/514b64ca-86fe-472f-84c2-f6e73e0ba300" />

# EVALUATION METRICS
<img width="1042" height="37" alt="image" src="https://github.com/user-attachments/assets/88d0e44c-0a28-4dd9-ace2-8c1543399076" />

# Confusion Matrix
<img width="1042" height="67" alt="image" src="https://github.com/user-attachments/assets/2368b6b0-1641-475e-ac3b-321a7ed8ebc9" />

# CLASSIFICATION REPORT

<img width="1042" height="400" alt="image" src="https://github.com/user-attachments/assets/5d867b53-879a-41a1-a13f-8b4f0c2469f1" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
