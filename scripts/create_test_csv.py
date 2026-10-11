
import pandas as pd
from pathlib import Path

dataset_folder = Path("dataset/MachineLearningCVE")

files = {
    "BENIGN": "Monday-WorkingHours.pcap_ISCX.csv",
    "DDoS": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "PortScan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
}

samples = []

for label, filename in files.items():
    file_path = dataset_folder / filename

    print("Reading:", filename)

    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    selected = df[df["Label"] == label].head(100)

    if selected.empty:
        print("No matching rows found for:", label)
    else:
        samples.append(selected)

if not samples:
    raise ValueError("No matching records were found.")

test_df = pd.concat(samples, ignore_index=True)

test_df = test_df.replace([float("inf"), float("-inf")], float("nan"))
test_df = test_df.dropna()

output_folder = Path("uploads")
output_folder.mkdir(exist_ok=True)

output_path = output_folder / "real_cicids_test.csv"
test_df.to_csv(output_path, index=False)

print("\nTest CSV created:", output_path)
print("\nActual labels:")
print(test_df["Label"].value_counts())
print("\nFinal shape:", test_df.shape)
