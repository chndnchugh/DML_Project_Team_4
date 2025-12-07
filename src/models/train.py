import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.config import load_config, PROJECT_ROOT
from src.data.preprocess import preprocess
from src.utils import ensure_dir


def train():
    cfg = load_config()

    # Set MLflow tracking URI to local directory
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")
    mlflow.set_experiment("kidney_mlops")



    X_train, X_test, y_train, y_test = preprocess()

    rf_params = cfg["model"]["rf_params"]
    model = RandomForestClassifier(**rf_params)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(rf_params)
        mlflow.log_metric("accuracy", acc)

        ensure_dir(PROJECT_ROOT / "models/model.pkl")
        joblib.dump(model, PROJECT_ROOT / "models/model.pkl")

        mlflow.sklearn.log_model(model, "model")

        print(f"Training complete. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train()
