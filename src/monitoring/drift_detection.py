"""
Local Monitoring using Evidently

This module provides data drift detection and model monitoring capabilities
using the Evidently library. It generates reports comparing reference (training)
data with current (production) data to detect distribution shifts.

Compatible with Evidently 0.7+
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.config import load_config, PROJECT_ROOT


def load_reference_data() -> pd.DataFrame:
    """
    Load the reference (training) dataset.
    """
    cfg = load_config()
    raw_path = PROJECT_ROOT / cfg["data"]["raw_path"]
    df = pd.read_csv(raw_path)
    return df


def get_data_definition() -> DataDefinition:
    """
    Create data definition for Evidently based on config.
    """
    cfg = load_config()

    return DataDefinition(
        numerical_columns=cfg["features"]["numeric"],
        categorical_columns=cfg["features"]["categorical"] + [cfg["data"]["target"]],
    )


def generate_drift_report(
    current_data: pd.DataFrame = None,
    reference_data: pd.DataFrame = None,
    output_path: str = None
) -> dict:
    """
    Generate a data drift report comparing reference and current data.

    Args:
        current_data: Current/production data to analyze. If None, uses a sample from reference.
        reference_data: Reference/training data. If None, loads from raw data path.
        output_path: Path to save HTML report. If None, saves to reports/drift_report.html

    Returns:
        dict: Summary of drift detection results
    """
    cfg = load_config()

    # Load reference data if not provided
    if reference_data is None:
        reference_data = load_reference_data()

    # If no current data, simulate by taking a sample (for demo purposes)
    if current_data is None:
        # Simulate production data by sampling from reference with slight modifications
        current_data = reference_data.sample(frac=0.3, random_state=42).copy()
        # Add some noise to simulate drift
        numeric_cols = cfg["features"]["numeric"]
        for col in numeric_cols:
            if col in current_data.columns:
                current_data[col] = current_data[col] * 1.05  # 5% shift

    data_definition = get_data_definition()

    # Create datasets
    reference_dataset = Dataset.from_pandas(
        reference_data,
        data_definition=data_definition
    )
    current_dataset = Dataset.from_pandas(
        current_data,
        data_definition=data_definition
    )

    # Create drift report
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    # Run the report - returns a Run object
    run_result = drift_report.run(
        reference_data=reference_dataset,
        current_data=current_dataset,
    )

    # Save HTML report
    if output_path is None:
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "drift_report.html"

    run_result.save_html(str(output_path))
    print(f"Drift report saved to: {output_path}")

    # Extract summary metrics
    report_dict = run_result.dict()

    summary = {
        "timestamp": datetime.now().isoformat(),
        "reference_samples": len(reference_data),
        "current_samples": len(current_data),
        "report_path": str(output_path),
    }

    # Extract drift detection results from the report
    metric_results = report_dict.get("metric_results", [])
    for metric in metric_results:
        metric_id = str(metric.get("metric", ""))
        result = metric.get("result", {})

        if "DatasetDrift" in metric_id or "dataset_drift" in str(result):
            summary["dataset_drift"] = result.get("dataset_drift", False)
            summary["drift_share"] = result.get("drift_share", 0.0)
            summary["number_of_drifted_columns"] = result.get("number_of_drifted_columns", 0)

    return summary


def generate_data_summary_report(
    data: pd.DataFrame = None,
    output_path: str = None
) -> dict:
    """
    Generate a data summary report for the given dataset.

    Args:
        data: Dataset to analyze. If None, loads reference data.
        output_path: Path to save HTML report.

    Returns:
        dict: Summary of data quality metrics
    """
    if data is None:
        data = load_reference_data()

    data_definition = get_data_definition()

    current_dataset = Dataset.from_pandas(
        data,
        data_definition=data_definition
    )

    summary_report = Report(metrics=[
        DataSummaryPreset(),
    ])

    run_result = summary_report.run(
        reference_data=None,
        current_data=current_dataset,
    )

    # Save HTML report
    if output_path is None:
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "data_summary_report.html"

    run_result.save_html(str(output_path))
    print(f"Data summary report saved to: {output_path}")

    return {
        "timestamp": datetime.now().isoformat(),
        "samples": len(data),
        "report_path": str(output_path),
    }


def generate_full_monitoring_report(
    current_data: pd.DataFrame = None,
    reference_data: pd.DataFrame = None
) -> dict:
    """
    Generate comprehensive monitoring reports including data drift
    and data summary.

    Args:
        current_data: Current/production data
        reference_data: Reference/training data

    Returns:
        dict: Combined summary of all monitoring reports
    """
    print("=" * 60)
    print("Generating Comprehensive Monitoring Reports")
    print("=" * 60)

    if reference_data is None:
        reference_data = load_reference_data()

    if current_data is None:
        current_data = reference_data.sample(frac=0.3, random_state=42).copy()
        cfg = load_config()
        numeric_cols = cfg["features"]["numeric"]
        for col in numeric_cols:
            if col in current_data.columns:
                current_data[col] = current_data[col] * 1.05

    results = {
        "timestamp": datetime.now().isoformat(),
    }

    # Data Drift Report
    print("\n1. Generating Data Drift Report...")
    drift_summary = generate_drift_report(current_data, reference_data)
    results["data_drift"] = drift_summary

    # Data Summary Report
    print("\n2. Generating Data Summary Report...")
    summary_result = generate_data_summary_report(current_data)
    results["data_summary"] = summary_result

    # Save combined summary
    reports_dir = PROJECT_ROOT / "reports"
    summary_path = reports_dir / "monitoring_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print("Monitoring Complete!")
    print(f"Summary saved to: {summary_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # Run full monitoring report
    summary = generate_full_monitoring_report()

    print("\nMonitoring Summary:")
    print(f"  - Dataset Drift Detected: {summary.get('data_drift', {}).get('dataset_drift', 'N/A')}")
    print(f"  - Drift Share: {summary.get('data_drift', {}).get('drift_share', 'N/A')}")
    print(f"  - Drifted Columns: {summary.get('data_drift', {}).get('number_of_drifted_columns', 'N/A')}")
