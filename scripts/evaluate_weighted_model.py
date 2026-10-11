
import os
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# File paths
DATASET_PATH = os.path.join(
    "dataset",
    "MachineLearningCVE",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

MODEL_PATH = os.path.join(
    "models",
    "random_forest_weighted_model.pkl"
)

print("Loading weighted model...")
model = joblib.load(MODEL_PATH)
print("Weighted model loaded successfully!")

print("\nLoading unseen DDoS dataset...")
data = pd.read_csv(DATASET_PATH, low_memory=False)
data.columns = data.columns.str.strip()
data["Label"] = data["Label"].astype(str).str.strip()

print("Original dataset shape:", data.shape)

# Clean invalid values
data.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Create actual binary labels
y_test = data["Label"].apply(
    lambda label: "Normal" if label == "BENIGN" else "Attack"
)

# Prepare features in the same order as the model
X_test = data.drop(columns=["Label"])
X_test = X_test.apply(pd.to_numeric, errors="coerce")

# Keep only rows with valid numeric features
valid_rows = X_test.notna().all(axis=1)
X_test = X_test.loc[valid_rows]
y_test = y_test.loc[valid_rows]

# Ensure the model receives the same features in the same order
feature_names = list(model.feature_names_in_)

missing_features = set(feature_names) - set(X_test.columns)
if missing_features:
    raise ValueError(f"Missing model features: {missing_features}")

X_test = X_test[feature_names]

print("\nCleaned test dataset shape:", X_test.shape)
print("\nActual target distribution:")
print(y_test.value_counts())

print("\nPredicting unseen DDoS traffic...")
y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["Normal", "Attack"],
        zero_division=0
    )
)

print("\nConfusion matrix (Normal, Attack):")
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Normal", "Attack"]
)
print(cm)

# Count missed attacks and false alarms
missed_attacks = ((y_test == "Attack") & (y_pred == "Normal")).sum()
detected_attacks = ((y_test == "Attack") & (y_pred == "Attack")).sum()
false_alarms = ((y_test == "Normal") & (y_pred == "Attack")).sum()

print("\nUnseen DDoS evaluation summary:")
print("Detected attacks:", detected_attacks)
print("Missed attacks:", missed_attacks)
print("False alarms:", false_alarms)

print("\nWeighted model evaluation completed!")
