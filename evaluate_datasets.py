import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import(
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
classification_report
)

model = joblib.load("models/random_forest_model.pkl")

result = []

print("Random Forest model loaded successfully!")

def evaluate_dataset(file_path):

    print("\nLoading dataset")

    data = pd.read_csv(file_path)

    data.columns = data.columns.str.strip()

    data.replace([np.inf, -np.inf], np.nan, inplace = True)
    data.dropna(inplace=True)

    actual_labels = data["Label"].str.strip()

    y_true = actual_labels.apply(
        lambda x: "Normal" if x == "BENIGN" else "Attack"
    )

    X = data.drop(columns=["Label"])

    expected_features = model.feature_names_in_

    X = X[list(expected_features)]

    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, pos_label="Attack", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, pos_label="Attack", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, pos_label="Attack", zero_division=0
    )

    result.append({
        "Dataset": os.path.basename(file_path),
        "Rows Evaluated": len(y_true),
        "Accuracy": accuracy,
        "Prescision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    print("\nResults for:", file_path)
    print("Rows evaluated:", len(y_true))
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=["Normal", "Attack"]
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=["Normal", "Attack"],
            zero_division = 0
        )
    )

dataset_folder = "dataset/MachineLearningCVE"

files = [
        file for file in os.listdir(dataset_folder)
        if file.endswith(".csv")
    ]

print("\nTotal datasets found:", len(files))

for file in files:
    file_path = os.path.join(dataset_folder, file)

    evaluate_dataset((file_path))

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(result)

results_df.to_csv(
    "results/dataset_evaluation.csv",
    index=False
)

print("\nEvaluation results saved to results/dataset_evaluation.csv")
