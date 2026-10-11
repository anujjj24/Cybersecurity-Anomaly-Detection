import os
import pandas as pd

os.makedirs("results", exist_ok=True)

comparison = {
    "Metric": [
        "Accuracy (%)",
        "Detected Attacks",
        "Missed Attacks",
        "False Alarms",
        "Attack Recall (%)"
    ],
    "Previous Model": [
        78.9432,
        81118,
        46896,
        78,
        63.36
    ],
    "Weighted Model": [
        78.9275,
        81171,
        46843,
        166,
        63.41
    ]
}

df = pd.DataFrame(comparison)

output_path = os.path.join("results", "model_comparison.csv")
df.to_csv(output_path, index=False)

print("Model comparison saved successfully!")
print(df)
print("File location:", output_path)