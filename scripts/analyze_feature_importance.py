import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/random_forest_unseen_model.pkl")

importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features:")
print(importance.head(20).to_string(index=False))

importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

top_features = importance.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(12, 8))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Network Features")
plt.title("Top 20 Features - Experimental Random Forest")
plt.tight_layout()

plt.savefig("results/feature_importance.png", dpi=300)
plt.show()

print("\nSaved results/features_importance.csv")
print("Saved results/feature_importance.png")