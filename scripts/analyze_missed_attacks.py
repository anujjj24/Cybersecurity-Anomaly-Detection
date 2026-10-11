import os
import pandas as pd
import numpy as np
import joblib

dataset_path = "dataset/MachineLearningCVE"
unseen_file = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
model_path = "models/random_forest_unseen_model.pkl"

model = joblib.load(model_path)
df = pd.read_csv(os.path.join(dataset_path, unseen_file))

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df["Label"] = df["Label"].str.strip()
df["Target"] = df["Label"].apply(
    lambda x: "Normal" if x == "BENIGN" else  "Attack"
)

X_test = df.drop(columns=["Label", "Target"]).copy()
X_test = X_test.loc[:, ~X_test.columns.duplicated()]
X_test = X_test.apply(pd.to_numeric, errors = "coerce" )
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

valid_rows = X_test.notna().all(axis=1)
X_test = X_test.loc[valid_rows].copy()
y_test = df.loc[X_test.index, "Target"]

X_test = X_test.loc[:, model.feature_names_in_]

y_pred = model.predict(X_test)

missed = (y_test == "Attack") & (y_pred == "Normal")
detected = (y_test == "Attack") & (y_pred == "Attack")
normal = y_test == "Normal"

groups = {
    "Missed Attacks": X_test.loc[missed],
    "Detected Attacks": X_test.loc[detected],
    "Normal Attacks": X_test.loc[normal]
}

features = [
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

report = pd.DataFrame({
    name: group[features].mean()
    for name , group in groups.items()
    if len(group) > 0
})

os.makedirs("results", exist_ok=True)
report.to_csv("results/missed_attacks_comparison.csv")

print("\nFeature comparison report:")
print(report.to_string())

print("\nGroup sizes:")
for name, group in groups.items():
    print(f"{name}: {len(group)}")

print("\nSaved results/missed_attacks_comparison.csv")