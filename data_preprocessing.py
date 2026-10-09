import pandas as pd
import os
import numpy as np


def prepare_data():

    dataset_path = "dataset/MachineLearningCVE"

    files = os.listdir(dataset_path)

    dataframes = []

    for file in files:

        if file.endswith(".csv"):

            file_path = os.path.join(dataset_path, file)

            print("Loading:", file)

            df = pd.read_csv(file_path)

            df["Source_File"] = file

            dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)

    data.columns = data.columns.str.strip()

    print("\nTotal shape:")
    print(data.shape)

    print("\nMissing values:")
    missing_values = data.isnull().sum()

    print(missing_values[missing_values > 0])

    numeric_data = data.select_dtypes(include=np.number)

    infinite_values = np.isinf(numeric_data).sum()

    print("\nInfinite values:")
    print(infinite_values[infinite_values > 0])

    print("\nDuplicate rows:")
    print(data.duplicated().sum())

    print("\nData type summary:")
    print(data.dtypes.value_counts())

    print("\nLabels:")
    print(data["Label"].value_counts())

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("\nInfinite values after replacement:")

    numeric_data = data.select_dtypes(include=np.number)

    infinite_values = np.isinf(numeric_data).sum()

    print(infinite_values[infinite_values > 0])

    print("\nMissing values after replacing infinity:")

    missing_values = data.isnull().sum()

    print(missing_values[missing_values > 0])

    data = data.dropna()

    print("\nShape after removing missing values:")
    print(data.shape)

    print("\nDuplicate rows after removing missing values:")
    print(data.duplicated().sum())

    print("\nRemoving duplicate rows...")

    data = data.drop_duplicates()

    print("\nCleaning labels...")

    data["Label"] = data["Label"].str.strip()

    data["Target"] = data["Label"].apply(
        lambda x: "Normal" if x == "BENIGN" else "Attack"
    )

    print("\nTarget distribution:")
    print(data["Target"].value_counts())

    print("Shape after removing duplicates:")
    print(data.shape)

    print("\n--- Final Cleaning check ---")

    print("\nMissing values:")
    print(data.isnull().sum().sum())

    numeric_data = data.select_dtypes(include=np.number)

    print("\nInfinite values:")
    print(np.isinf(numeric_data).sum().sum())

    print("\nDuplicate rows:")
    print(data.duplicated().sum())

    print("\nFinal shape:")
    print(data.shape)

    X = data.drop(columns=["Label", "Target", "Source_File"])
    y = data["Target"]

    return X, y


if __name__ == "__main__":
    prepare_data()