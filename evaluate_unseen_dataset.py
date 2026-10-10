
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

X_test = df.drop(columns=["Label", "Target"])

print("\nChecking feature compatibility...")

print("Number of test features:", X_test.shape[1])
print("Number of model features:", model.n_features_in_)

print(
    "Feature names match:",
    list(X_test.columns) == list(model.feature_names_in_)
)

y_test = df["Target"]

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