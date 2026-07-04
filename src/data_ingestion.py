import pandas as pd
from pathlib import Path

from src.config import load_config


def load_dataset():
    """
    Load the dataset specified in the configuration file.

    Returns:
        pd.DataFrame: Loaded dataset
    """

    config = load_config()

    dataset_path = Path(config["dataset"]["raw_data_path"])

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    return df


def validate_schema(df):
    """
    Validate whether all required columns are present.
    """

    required_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )


def validate_dataset(df):
    """
    Perform basic dataset validation.
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    duplicate_rows = df.duplicated().sum()

    if duplicate_rows > 0:
        print(f"Warning: {duplicate_rows} duplicate rows found.")

    missing_values = df.isnull().sum()

    if missing_values.sum() > 0:
        print("\nMissing Values:")
        print(missing_values[missing_values > 0])

    print("\nDataset Validation Successful.")


def get_data():
    """
    Load and validate dataset.

    Returns:
        pd.DataFrame
    """

    df = load_dataset()

    validate_schema(df)

    validate_dataset(df)

    return df


if __name__ == "__main__":

    df = get_data()

    print(df.head())

    print("\nShape:", df.shape)