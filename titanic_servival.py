# Titanic Survival Prediction Model
# Save this as titanic_model.py and run in VS Code (Python 3.8+)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("Titanic-Dataset (2).csv")  # make sure CSV is in same folder

# =======================
# Data Preprocessing
# =======================
# Drop columns not useful for prediction
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Fill missing Age values with median
data["Age"].fillna(data["Age"].median(), inplace=True)

# Fill missing Embarked values with mode
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Convert categorical variables to numeric
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])       # male=1, female=0
data["Embarked"] = le.fit_transform(data["Embarked"])

# Features & Target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# =======================
# Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# Model Training
# =======================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# =======================
# Model Evaluation
# =======================
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
