import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

from src.data_ingestion import get_data


def run_drift_detection():

    # Reference dataset
    reference_data = get_data()

    # Simulate production dataset
    current_data = reference_data.sample(frac=0.3, random_state=42).copy()

    # Introduce drift for demonstration
    current_data["MonthlyCharges"] *= 1.25

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
        )

    my_eval = report.run(
        current_data=current_data,
        reference_data=reference_data
        )

    my_eval.save_html("monitoring/drift_report.html")

    print("Drift report generated successfully.")


if __name__ == "__main__":
    run_drift_detection()