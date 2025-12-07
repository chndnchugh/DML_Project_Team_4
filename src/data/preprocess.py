import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.config import load_config, PROJECT_ROOT
from src.utils import ensure_dir
from src.data.load_data import load_raw_data


def preprocess():
    cfg = load_config()
    df = load_raw_data()

    target = cfg["data"]["target"]
    y = df[target]
    X = df.drop(columns=[target])

    num_cols = cfg["features"]["numeric"]
    cat_cols = cfg["features"]["categorical"]

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna("Unknown")

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    processed_path = PROJECT_ROOT / cfg["data"]["processed_path"]
    ensure_dir(processed_path)

    X_train_df = pd.DataFrame(
        X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed
    )
    X_train_df[target] = y_train.values
    X_train_df.to_csv(processed_path, index=False)

    ensure_dir(PROJECT_ROOT / "models/preprocessor.pkl")
    joblib.dump(preprocessor, PROJECT_ROOT / "models/preprocessor.pkl")

    print("Preprocessing complete.")
    print("Train shape:", X_train_transformed.shape)
    print("Test shape:", X_test_transformed.shape)

    return X_train_transformed, X_test_transformed, y_train, y_test


if __name__ == "__main__":
    preprocess()
