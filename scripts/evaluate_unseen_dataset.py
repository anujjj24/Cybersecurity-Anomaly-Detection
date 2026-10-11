
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

dataset_path = "dataset/MachineLearningCVE"
unseen_file = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

file_path = os.path.join(dataset_path, unseen_file)

model_path = "models/random_forest_unseen_model.pkl"

print("Loading experimental model...")

model = joblib.load(model_path)

print("Experimental model loaded successfully!")

print("\nLoading unseen DDoS dataset...")

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print("Original dataset shape:", df.shape)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df["Label"] = df["Label"].str.strip()

print("\nOriginal attack label distribution:")
print(df["Label"].value_counts())

df["Target"] = df["Label"].apply(
    lambda x: "Normal" if x == "BENIGN" else "Attack"
)

X_test = df.drop(columns=["Label", "Target"]).copy()

# Remove duplicate feature columns
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

# Convert features to numeric
X_test = X_test.apply(pd.to_numeric, errors="coerce")

# Remove rows with invalid numeric values
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
valid_rows = X_test.notna().all(axis=1)

X_test = X_test.loc[valid_rows].copy()
y_test = df.loc[X_test.index, "Target"]

print("\nChecking feature compatibility...")

if list(X_test.columns) != list(model.feature_names_in_):
    raise ValueError(
        "Test feature names or order do not match the trained model."
    )

print("Number of test features:", X_test.shape[1])
print("Number of model features:", model.n_features_in_)

print(
    "Feature names match:",
    list(X_test.columns) == list(model.feature_names_in_)
)

print("\nTest dataset shape:", X_test.shape)

print("\nActual target distribution:")
print(y_test.value_counts())

print("\nPredicting unseen dataset...")

y_pred = model.predict(X_test)

# Find DDoS records incorrectly predicted as Normal
missed_indices = X_test.index[
    (y_test == "Attack") & (y_pred == "Normal")
]

# Find DDoS records correctly predicted as Attack
detected_indices = X_test.index[
    (y_test == "Attack") & (y_pred == "Attack")
]

print("\nNumber of missed attacks:", len(missed_indices))
print("Number of detected attacks:", len(detected_indices))

# Compare selected feature averages
features_to_compare = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets"
]

comparison = pd.DataFrame({
    "Missed Attacks": X_test.loc[
        missed_indices, features_to_compare
    ].mean(),
    "Detected Attacks": X_test.loc[
        detected_indices, features_to_compare
    ].mean()
})

print("\nMissed vs Detected DDoS Feature Averages:")
print(comparison)

# Count the model's predictions
print("\nPrediction distribution:")
print(pd.Series(y_pred).value_counts())

# Compare actual labels with predicted labels
print("\nActual vs Predicted labels:")
print(pd.crosstab(
    y_test,
    y_pred,
    rownames=["Actual"],
    colnames=["Predicted"]
))

accuracy = accuracy_score(y_test, y_pred)

print("\nUnseen DDoS Dataset Accuracy:")
print(accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Normal", "Attack"]
))

# Analyze correctly detected and missed DDoS attacks

X_test_analysis = X_test.copy()

X_test_analysis["Actual"] = y_test
X_test_analysis["Predicted"] = y_pred

missed_attacks = X_test_analysis[
    (X_test_analysis["Actual"] == "Attack") &
    (X_test_analysis["Predicted"] == "Normal")
]

detected_attacks = X_test_analysis[
    (X_test_analysis["Actual"] == "Attack") &
    (X_test_analysis["Predicted"] == "Attack")
]

normal_traffic = X_test_analysis[
    X_test_analysis["Actual"] == "Normal"
]

important_features = [
    "Packet Length Std",
    "Packet Length Variance",
    "Max Packet Length",
    "Avg Bwd Segment Size",
    "Bwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Subflow Fwd Bytes",
    "Average Packet Size",
    "Idle Mean"
]

print("\n=== Average Feature Values ===")

for feature in important_features:
    if feature in X_test_analysis.columns:
        print(f"\n{feature}")
        print("Missed attacks:", missed_attacks[feature].mean())
        print("Detected attacks:", detected_attacks[feature].mean())
        print("Normal traffic:", normal_traffic[feature].mean())

print("\nFeature comparison completed!")