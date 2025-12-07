"""
Prefect Pipeline Orchestration for Kidney MLOps Project

This module defines the ML pipeline using Prefect for workflow orchestration.
It coordinates data preprocessing, model training, and evaluation tasks.
"""

from prefect import flow, task
from prefect.logging import get_run_logger

from src.data.preprocess import preprocess
from src.models.train import train
from src.models.evaluate import evaluate


@task(name="preprocess_data", retries=2, retry_delay_seconds=10)
def preprocess_task():
    """
    Task to preprocess raw kidney dataset.
    Handles missing values, scaling, and encoding.
    """
    logger = get_run_logger()
    logger.info("Starting data preprocessing...")

    X_train, X_test, y_train, y_test = preprocess()

    logger.info(f"Preprocessing complete. Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


@task(name="train_model", retries=1)
def train_task():
    """
    Task to train RandomForest model with MLflow tracking.
    """
    logger = get_run_logger()
    logger.info("Starting model training with MLflow tracking...")

    train()

    logger.info("Model training complete.")


@task(name="evaluate_model")
def evaluate_task():
    """
    Task to evaluate the trained model and save metrics.
    """
    logger = get_run_logger()
    logger.info("Starting model evaluation...")

    evaluate()

    logger.info("Model evaluation complete. Metrics saved to reports/metrics.json")


@flow(name="kidney_mlops_pipeline", description="End-to-end ML pipeline for CKD prediction")
def kidney_ml_pipeline(run_preprocessing: bool = True, run_training: bool = True, run_evaluation: bool = True):
    """
    Main Prefect flow that orchestrates the complete ML pipeline.

    Args:
        run_preprocessing: Whether to run data preprocessing step
        run_training: Whether to run model training step
        run_evaluation: Whether to run model evaluation step

    Returns:
        dict: Pipeline execution status
    """
    logger = get_run_logger()
    logger.info("Starting Kidney MLOps Pipeline...")

    results = {
        "preprocessing": False,
        "training": False,
        "evaluation": False
    }

    if run_preprocessing:
        preprocess_task()
        results["preprocessing"] = True

    if run_training:
        train_task()
        results["training"] = True

    if run_evaluation:
        evaluate_task()
        results["evaluation"] = True

    logger.info("Pipeline execution complete!")
    return results


@flow(name="training_only_pipeline", description="Quick pipeline for model retraining")
def training_pipeline():
    """
    Simplified flow for retraining the model without preprocessing.
    Assumes preprocessed data already exists.
    """
    logger = get_run_logger()
    logger.info("Starting training-only pipeline...")

    train_task()
    evaluate_task()

    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    # Run the full pipeline
    kidney_ml_pipeline()
