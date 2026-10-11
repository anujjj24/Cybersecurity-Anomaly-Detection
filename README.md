# Cybersecurity Anomaly Detection

A machine-learning project that analyzes network traffic CSV files and classifies network flows as **Normal** or **Attack** using a Random Forest classifier.

## Features

- Network traffic CSV upload
- Machine-learning-based anomaly classification
- Dashboard displaying normal traffic, detected attacks, and attack rate
- Traffic distribution visualization
- Downloadable prediction results
- Dataset evaluation and feature-importance analysis
- Comparison of standard and class-weighted Random Forest models

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- Matplotlib and Seaborn
- Chart.js
- CICIDS2017 dataset

## Project Structure

```text
Cybersecurity-Anomaly-Detection1/
├── app.py
├── src/
│   ├── data_preprocessing.py
│   └── train_model.py
├── scripts/
│   ├── analyze_feature_importance.py
│   ├── analyze_missed_attacks.py
│   ├── create_model_comparison.py
│   ├── create_test_csv.py
│   ├── evaluate_datasets.py
│   ├── evaluate_unseen_dataset.py
│   ├── evaluate_weighted_model.py
│   ├── train_unseen_model.py
│   ├── train_weighted_model.py
│   └── visualize_results.py
├── dataset/
├── models/
├── results/
├── templates/
│   └── index.html
├── uploads/
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## Setup

1. Install Python and Git.
2. Clone this repository.
3. Open the project folder in PyCharm.
4. Install the required Python dependencies.
5. Run `app.py`.
6. Open `http://127.0.0.1:5000` in your browser.
7. Upload a compatible network traffic CSV file.

## Dataset

This project uses the CICIDS2017 network intrusion detection dataset. The dataset contains benign traffic and multiple types of attacks.

The dataset files are not included in this repository. Obtain the dataset separately and place the CSV files in the expected dataset directory.

## Model Evaluation

The project includes a standard Random Forest model and experimental models trained with different approaches.

The original model achieved approximately 99.85% accuracy on its random test split. However, on a held-out DDoS dataset from CICIDS2017, attack recall was approximately 63.36%. The class-weighted experiment produced only a small improvement in attack recall and increased false alarms.

These results demonstrate why random-split accuracy alone is not sufficient to evaluate intrusion detection performance.

## Limitations

- The application analyzes uploaded CSV files rather than monitoring live network traffic.
- Detection performance depends on the training data and input features.
- Some attacks may be missed, and some normal traffic may be flagged as malicious.
- Evaluation on an independent real-world dataset is still needed.

## Future Improvements

- Evaluate the model on independent datasets.
- Improve attack recall while controlling false alarms.
- Add more comprehensive automated tests.
- Improve validation and error handling for uploaded CSV files.
- Explore real-time monitoring as a separate future enhancement.

## Disclaimer

This project is intended for educational and research purposes. It is not a replacement for a production-grade