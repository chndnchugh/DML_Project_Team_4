from pathlib import Path

def ensure_dir(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
