
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("models/random_forest_model.pkl")

# Create required folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "file" not in request.files:
        return render_template(
            "index.html", error="No file was uploaded."
        )

    file = request.files["file"]

    if file.filename == "":
        return render_template(
            "index.html", error="Please select a CSV file."
        )

    if not file.filename.lower().endswith(".csv"):
        return render_template(
            "index.html", error="Please upload a CSV file."
        )

    filename = secure_filename(file.filename)

    if not filename:
        return render_template(
            "index.html", error="The uploaded filename is invalid."
        )

    file_path = os.path.join("uploads", filename)

    try:
        file.save(file_path)

        original_data = pd.read_csv(file_path)
        original_data.columns = original_data.columns.str.strip()

        if original_data.empty:
            return render_template(
                "index.html", error="The uploaded CSV is empty."
            )

        # Preserve ground-truth labels if provided
        actual_labels = None

        if "Label" in original_data.columns:
            actual_labels = (
                original_data["Label"].astype(str).str.strip()
            )

        # Remove labels before prediction
        data = original_data.drop(
            columns=["Label"], errors="ignore"
        ).copy()

        # Remove duplicate column names
        data = data.loc[:, ~data.columns.duplicated()]

        expected_features = list(model.feature_names_in_)

        missing_features = [
            col for col in expected_features
            if col not in data.columns
        ]

        if missing_features:
            return render_template(
                "index.html",
                error=(
                    "Missing required features: "
                    + ", ".join(missing_features)
                )
            )

        # Match the model's training feature order
        data = data[expected_features]

        # Clean invalid numeric values
        data = data.apply(pd.to_numeric, errors="coerce")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        valid_rows = data.notna().all(axis=1)
        data = data.loc[valid_rows].copy()

        if data.empty:
            return render_template(
                "index.html",
                error="No valid numeric rows were found."
            )

        if actual_labels is not None:
            actual_labels = actual_labels.loc[data.index]

        # Make predictions
        predictions = model.predict(data)

        normal_count = int((predictions == "Normal").sum())
        attack_count = int((predictions == "Attack").sum())
        total_count = len(predictions)

        attack_percentage = (
            attack_count / total_count * 100
        )

        # Create row-by-row results
        results = pd.DataFrame({
            "Prediction": predictions
        }, index=data.index)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(data)
            class_names = list(model.classes_)

            if "Attack" in class_names:
                attack_index = class_names.index("Attack")
                results["Attack Probability (%)"] = (
                    probabilities[:, attack_index] * 100
                ).round(2)

        accuracy = None

        if actual_labels is not None:
            results["Actual Label"] = actual_labels

            normalized = actual_labels.str.lower().str.strip()

            actual_binary = normalized.map({
                "benign": "Normal",
                "normal": "Normal",
                "attack": "Attack",
                "ddos": "Attack",
                "portscan": "Attack",
                "dos hulk": "Attack",
                "dos goldeneye": "Attack",
                "dos slowloris": "Attack",
                "dos slowhttptest": "Attack",
                "ftp-patator": "Attack",
                "ssh-patator": "Attack",
                "bot": "Attack",
                "infiltration": "Attack",
                "heartbleed": "Attack",
                "web attack - brute force": "Attack",
                "web attack - xss": "Attack",
                "web attack - sql injection": "Attack"
            })

            comparable = actual_binary.notna()

            if comparable.any():
                correct = (
                    actual_binary.loc[comparable]
                    == pd.Series(
                        predictions, index=data.index
                    ).loc[comparable]
                )

                accuracy = correct.mean() * 100

                results["Correct"] = ""
                results.loc[comparable, "Correct"] = correct.map({
                    True: "Yes",
                    False: "No"
                })

        # Save the complete results for download
        result_filename = (
            os.path.splitext(filename)[0] + "_results.csv"
        )
        result_path = os.path.join("results", result_filename)
        results.reset_index(drop=True).to_csv(
            result_path, index=False
        )

        # Show at most 100 rows in the dashboard
        table_rows = (
            results.head(100)
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

        print("File received:", filename)
        print("Rows analyzed:", total_count)
        print("Normal:", normal_count)
        print("Attacks:", attack_count)

        return render_template(
            "index.html",
            filename=filename,
            total_count=total_count,
            normal_count=normal_count,
            attack_count=attack_count,
            attack_percentage=round(attack_percentage, 2),
            accuracy=round(accuracy, 2) if accuracy is not None else None,
            table_rows=table_rows,
            result_count=len(results),
            download_ready=True,
            result_filename=result_filename
        )

    except pd.errors.EmptyDataError:
        return render_template(
            "index.html",
            error="The uploaded file contains no readable CSV data."
        )

    except pd.errors.ParserError:
        return render_template(
            "index.html",
            error="Could not parse the CSV. Check its format."
        )

    except Exception:
        app.logger.exception("Analysis failed")
        return render_template(
            "index.html",
            error="Analysis failed. Check the PyCharm console for details."
        )


@app.route("/download/<path:filename>")
def download_results(filename):
    safe_filename = secure_filename(filename)

    if safe_filename != filename or not safe_filename.endswith("_results.csv"):
        return "Invalid results filename.", 400

    return send_from_directory(
        os.path.abspath("results"),
        safe_filename,
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)
