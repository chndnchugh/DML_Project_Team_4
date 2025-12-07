import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import load_config, PROJECT_ROOT


def evaluate():
    cfg = load_config()
    target = cfg["data"]["target"]

    processed_path = PROJECT_ROOT / cfg["data"]["processed_path"]
    df = pd.read_csv(processed_path)

    X = df.drop(columns=[target])
    y_true = df[target]

    model = joblib.load(PROJECT_ROOT / "models/model.pkl")

    y_pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    evaluate()
