import pandas as pd
from src.config import load_config, PROJECT_ROOT

def load_raw_data() -> pd.DataFrame:
    cfg = load_config()
    raw_path = PROJECT_ROOT / cfg["data"]["raw_path"]
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")
    df = pd.read_csv(raw_path)
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(df.info())
