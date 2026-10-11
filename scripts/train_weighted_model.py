import pandas as pd
import os
import glob
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_model import y_val, X_val

DATASET_FOLDER = "dataset/MachineLearningCVE"
MODEL_FOLDER = "models"

UNSEEN_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

os.makedirs(MODEL_FOLDER, exist_ok=True)

print("Loading training datasets...")

dataframes = []

csv_files = glob.glob(os.path.join(DATASET_FOLDER, "*.csv"))

for file in csv_files:
    filename = os.path.basename(file)

    if filename == UNSEEN_FILE:
        print("Skipping unseen dataset:", filename)
        continue

    print("Loading:", filename)

    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()

    dataframes.append(df)

if not dataframes:
    raise ValueError("No training csv files were found.")

print("\ncombining datasets...")
data = pd.concat(dataframes, ignore_index=True)

data.columns = data.columns.str.strip()
data["Label"] = data["Label"].astype(str).str.strip()

data.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

data["Target"] = data["Label"].apply(
    lambda label: "Normal" if label == "BENIGN" else "Attack"
)

X = data.drop(columns=["Label", "Target"])
X = X.apply(pd.to_numeric, errors = "coerce")

valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows]
y = data.loc[valid_rows, "Target"]

print("\nDataset shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts())

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)


print("\nTraining class-weighted Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("\nValidation accuracy:")
print(accuracy_score(y_val, y_pred))

print("\nClassification report:")
print(classification_report(y_val, y_pred, labels=["Normal", "Attack"]))

print("\nConfusion matrix (Normal , Attack ):")
print(confusion_matrix(y_val, y_pred, labels=["Normal" , "Attack"]))

model_path = os.path.join(MODEL_FOLDER, "random_forest_weighted_model.pkl")
joblib.dump(model, model_path)

print("\nWeighted model saved successfully:")
print(model_path)