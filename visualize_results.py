import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("results/dataset_evaluation.csv")

data.columns = data.columns.str.strip()

data.rename(
    columns={
        "Prescision": "Precision",
        "F1_Score": "F1-Score"
    },
    inplace=True
)

print(data)
print(data.columns.tolist())

data["Short Name"] = [
    "Friday - DDoS",
    "Friday - PortScan",
    "Friday - Morning",
    "Monday",
    "Thursday - Infiltration",
    "Thursday - Web Attacks",
    "Tuesday",
    "Wednesday"
]

# Accuracy Comparison Chart
plt.figure(figsize=(12,6))

plt.bar(
    data["Short Name"],
    data["Accuracy"] * 100
)

plt.title("Random Forest Accuracy Across Eight Datasets")
plt.xlabel("Dataset")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("results/accuracy_comparison.png", dpi = 300)

plt.show()

print("Accuracy chart saved successfully!")

# Precision Comparison Chart
plt.figure(figsize=(12,6))

plt.bar(
    data["Short Name"],
    data["Precision"] * 100
)

plt.title("Random Forest Precision Across Eight Datasets")
plt.xlabel("Dataset")
plt.ylabel("Precision (%)")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("results/precision_comparison.png", dpi = 300)

plt.show()

print("Precision chart saved successfully!")

# Recall Comparison Chart
plt.figure(figsize=(12,6))
plt.bar(
    data["Short Name"],
    data["Recall"] * 100
)

plt.title("Random Forest Recall Across Eight Datasets")
plt.xlabel("Dataset")
plt.ylabel("Recall (%)")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("results/recall_comparison.png", dpi = 300)

plt.show()

print("Recall chart saved successfully!")

# F1-Score Comparison Chart

plt.figure(figsize=(12, 6))

plt.bar(
    data["Short Name"],
    data["F1-Score"] * 100
)

plt.title("Random Forest F1-Score Across Eight Datasets")
plt.xlabel("Dataset")
plt.ylabel("F1-Score (%)")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("results/f1_score_comparison.png", dpi=300)

plt.show()

print("F1-score chart saved successfully!")
