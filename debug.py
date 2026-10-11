
import pandas as pd
import joblib
from tkinter import Tk, filedialog

# Load the saved model
model = joblib.load("models/random_forest_model.pkl")

print("Model loaded successfully.")

# Select a CSV file
root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select the CSV you tested",
    filetypes=[("CSV files", "*.csv")]
)

root.destroy()

if not file_path:
    print("No file selected.")
    raise SystemExit

# Read the CSV
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

print("\nCSV shape:", df.shape)
print("CSV feature count:", len(model.feature_names_in_))

# Check for missing model features
missing_features = [
    col for col in model.feature_names_in_
    if col not in df.columns
]

print("\nMissing features:", missing_features)

if missing_features:
    print("This CSV does not contain all required model features.")
    raise SystemExit

# Use the exact training feature order
X = df.loc[:, model.feature_names_in_]

# Convert features to numeric values
X = X.apply(pd.to_numeric, errors="coerce")

print("\nRows:", len(X))
print("Rows with missing/non-numeric feature values:",
      X.isna().any(axis=1).sum())

# Replace invalid values safely for this diagnostic
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.dropna()

if X.empty:
    print("No valid rows remain after cleaning.")
    raise SystemExit

# Predict directly
predictions = model.predict(X)

print("\nPrediction counts:")
print(pd.Series(predictions).value_counts())

print("\nFirst 10 predictions:")
print(predictions[:10])

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)
    print("\nModel classes:", model.classes_)
    print("First 10 probability rows:")
    print(probabilities[:10])
