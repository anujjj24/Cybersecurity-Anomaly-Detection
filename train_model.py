from data_preprocessing import prepare_data
import time
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import(
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

X, y = prepare_data()

print("\nData loaded successfully")

print("Features shapes:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nData split successfully")

print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Validation target:", y_val.shape)
print("Testing target:", y_test.shape)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Starting model training...")

start_time = time.time()

model.fit(X_train, y_train)

end_time = time.time()

training_time = end_time - start_time

print("Model training completed!")
print("Training time:", training_time,"seconds")
print("Training time:", training_time/60, "minutes")

from sklearn.metrics import accuracy_score, classification_report

print("\nEvaluating model on validation data...")

y_val_pred = model.predict(X_val)

accuracy = accuracy_score(y_val , y_val_pred)

print("Validation accuracy :", accuracy)

print("\nClassification report:")
print(classification_report(y_val, y_val_pred))

print("\nEvaluating model on testing data...")

y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)

print("Test accuracy :", test_accuracy)

test_precision = precision_score(
    y_test, y_test_pred, pos_label="Attack"
)

test_recall = recall_score(
    y_test, y_test_pred, pos_label="Attack"
)

test_f1 = f1_score(
    y_test, y_test_pred, pos_label="Attack"
)

print("\nTest Evaluation Metrics:")
print(f"Accuracy: {test_accuracy:.6f}")
print(f"Precision: {test_precision:.6f}")
print(f"Recall: {test_recall:.6f}")
print(f"F1-score: {test_f1:.6f}")

print("\nTest Classification report:")
print(classification_report(y_test, y_test_pred))

labels = ["Normal", "Attack"]

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix = cm,
    display_labels= labels
)

display.plot(values_format="d")

plt.title("Random Forest - Test Confusion Matrix")
plt.tight_layout()

os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix saved to results/confusion_matrix.png")

print("\nSaving model...")

joblib.dump(model, "models/random_forest_model.pkl")

print("Model saved successfully!")