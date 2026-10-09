import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

dataset_path = "dataset/MachineLearningCVE"

unseen_file = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Dataset folder:", dataset_path)
print("File reserved for testing :", unseen_file)

dataframes = []

for file in os.listdir(dataset_path):

    if file.endswith(".csv") and file != unseen_file:

        file_path = os.path.join(dataset_path, file)

        print("Loading training dataset:", file)

        df = pd.read_csv(file_path)

        dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

print("\nCombined training data shape:")
print(data.shape)

data.columns = data.columns.str.strip()

data.replace([np.inf, -np.inf], np.nan, inplace=True)

data.dropna(inplace=True)

data.drop_duplicates(inplace=True)

data["Label"] = data["Label"].str.strip()

data["Target"] = data["Label"].apply(
    lambda x: "Normal" if x == "BENIGN" else "Attack"
)

print("\nTraining data shape after cleaning:")
print(data.shape)

print("\nTraining target distribution:")
print(data["Target"].value_counts())

X = data.drop(columns=["Label", "Target"])
y = data["Target"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining experimental Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print("\nValidation Accuracy:")
print(accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_unseen_model.pkl")

print("\nExperimental model saved successfully!")
