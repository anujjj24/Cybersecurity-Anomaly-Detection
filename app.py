from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("models/random_forest_model.pkl")

# Create the uploads folder if it does not exist
os.makedirs("uploads", exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    # Check whether a file was submitted
    if "file" not in request.files:
        return render_template(
            "index.html",
            error="No file was uploaded."
        )

    file = request.files["file"]

    # Check whether a file was selected
    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select a CSV file."
        )

    # Accept CSV files only
    if not file.filename.lower().endswith(".csv"):
        return render_template(
            "index.html",
            error="Please upload a CSV file."
        )

    # Secure the uploaded filename
    filename = secure_filename(file.filename)

    if not filename:
        return render_template(
            "index.html",
            error="The uploaded filename is invalid."
        )

    file_path = os.path.join("uploads", filename)

    try:
        # Save the uploaded file
        file.save(file_path)

        print("File received:", filename)
        print("File saved:", file_path)

        # Read the CSV file
        data = pd.read_csv(file_path)

        print("CSV loaded successfully")
        print("Original CSV shape:", data.shape)

        if data.empty:
            return render_template(
                "index.html",
                error="The uploaded CSV file is empty."
            )

        # Clean column names
        data.columns = data.columns.str.strip()

        # Remove infinite and missing values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        # Remove the original label column if present
        if "Label" in data.columns:
            data.drop(columns=["Label"], inplace=True)

        # Check that all expected model features exist
        expected_features = model.feature_names_in_

        missing_features = sorted(
            set(expected_features) - set(data.columns)
        )

        if missing_features:
            return render_template(
                "index.html",
                error=(
                    "This CSV is missing required network-flow features: "
                    + ", ".join(missing_features)
                )
            )

        # Remove additional columns the model does not use
        extra_features = set(data.columns) - set(expected_features)

        if extra_features:
            data.drop(columns=list(extra_features), inplace=True)

        # Arrange features in the exact order used during training
        data = data[list(expected_features)]

        if data.empty:
            return render_template(
                "index.html",
                error="No valid rows remain after cleaning the CSV."
            )

        # Check that all model features are numeric
        data = data.apply(pd.to_numeric, errors="coerce")

        # Remove rows that contain invalid numeric values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        if data.empty:
            return render_template(
                "index.html",
                error="No valid numeric network-flow rows were found."
            )

        # Predict using the trained Random Forest model
        prediction = model.predict(data)

        # Count normal traffic and attacks
        normal_count = int((prediction == "Normal").sum())
        attack_count = int((prediction == "Attack").sum())
        total_count = normal_count + attack_count

        print("Normal:", normal_count)
        print("Attack:", attack_count)
        print("Total analyzed:", total_count)

        # Display actual model predictions on the dashboard
        return render_template(
            "index.html",
            total_count=total_count,
            normal_count=normal_count,
            attack_count=attack_count,
            filename=filename
        )

    except pd.errors.EmptyDataError:
        return render_template(
            "index.html",
            error="The uploaded file does not contain readable CSV data."
        )

    except pd.errors.ParserError:
        return render_template(
            "index.html",
            error="The CSV file could not be parsed. Check its format."
        )

    except Exception as e:
        print("Error during analysis:", str(e))

        return render_template(
            "index.html",
            error=f"An error occurred during analysis: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)