# Kidney MLOps Project

An end-to-end MLOps project for **Chronic Kidney Disease (CKD) Prediction** using machine learning with complete pipeline orchestration, experiment tracking, monitoring, and deployment capabilities.

## Project Overview

This project demonstrates a production-ready ML pipeline that includes:
- **Data preprocessing** with DVC tracking
- **Model training** with MLflow experiment tracking
- **Pipeline orchestration** with Prefect
- **API deployment** with FastAPI
- **Containerization** with Docker
- **Data drift monitoring** with Evidently
- **CI/CD** with GitHub Actions

## Project Structure

```
DML_Project_Team_4/
├── .github/workflows/      # CI/CD GitHub Actions
│   └── ci.yml
├── app/                    # FastAPI Application
│   ├── main.py            # API endpoints
│   └── models/            # Trained model artifacts
├── src/                    # Source Code
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # Training & evaluation
│   ├── monitoring/        # Evidently drift detection
│   ├── pipeline.py        # Prefect orchestration
│   └── config.py          # Configuration loader
├── data/
│   ├── raw/               # Raw dataset
│   └── processed/         # Preprocessed data
├── notebooks/             # EDA notebooks
├── reports/               # Metrics & monitoring reports
├── tests/                 # Unit tests
├── models/                # Saved model artifacts
├── mlruns/                # MLflow experiment tracking
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── dvc.yaml               # DVC pipeline definition
├── params.yaml            # Model & data configuration
├── requirements.txt       # Python dependencies
└── pyproject.toml         # Project configuration
```

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/DML_Project_Team_4.git
cd DML_Project_Team_4
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Dataset

Place the Kaggle Kidney CKD dataset in:
```
data/raw/kidney_dataset.csv
```

## Running the Project

### Option 1: Run Individual Steps

```bash
# Step 1: Data Preprocessing
python -m src.data.preprocess

# Step 2: Model Training (with MLflow tracking)
python -m src.models.train

# Step 3: Model Evaluation
python -m src.models.evaluate
```

### Option 2: Run via DVC Pipeline

```bash
dvc repro
```

### Option 3: Run via Prefect Orchestration

```bash
python -m src.pipeline
```

## MLflow Experiment Tracking

View experiment runs and metrics:

```bash
mlflow ui
```

Open in browser: http://localhost:5000

## Data Drift Monitoring (Evidently)

Generate drift and data quality reports:

```bash
python -m src.monitoring.drift_detection
```

Reports are saved to:
- `reports/drift_report.html`
- `reports/data_summary_report.html`
- `reports/monitoring_summary.json`

## API Deployment

### Run Locally with FastAPI

```bash
uvicorn app.main:app --reload --port 8000
```

Access:
- Swagger UI: http://localhost:8000/docs
- API Root: http://localhost:8000

### Run with Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t kidney-mlops-api .
docker run -p 8000:10000 kidney-mlops-api
```

### Test the API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Creatinine": 1.2,
    "BUN": 15.0,
    "GFR": 90.0,
    "Urine_Output": 1500.0,
    "Age": 45.0,
    "Protein_in_Urine": 100.0,
    "Water_Intake": 2.5,
    "Diabetes": "No",
    "Hypertension": "No",
    "Medication": "None"
  }'
```

## Running Tests

```bash
pytest tests/ -v
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically runs on push:
- Linting with flake8
- Unit tests with pytest
- DVC pipeline validation
- Docker build verification
- Monitoring setup validation

## Commands Summary

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Preprocess data | `python -m src.data.preprocess` |
| Train model | `python -m src.models.train` |
| Evaluate model | `python -m src.models.evaluate` |
| Run DVC pipeline | `dvc repro` |
| Run Prefect pipeline | `python -m src.pipeline` |
| View MLflow UI | `mlflow ui` |
| Generate monitoring reports | `python -m src.monitoring.drift_detection` |
| Start API (local) | `uvicorn app.main:app --reload` |
| Start API (Docker) | `docker-compose up --build` |
| Run tests | `pytest tests/ -v` |

## Technologies Used

- **Python 3.10+**
- **scikit-learn** - Machine Learning
- **MLflow** - Experiment Tracking
- **DVC** - Data Version Control
- **Prefect** - Workflow Orchestration
- **FastAPI** - API Framework
- **Evidently** - Data Monitoring
- **Docker** - Containerization
- **GitHub Actions** - CI/CD

## Team

**Team 4** - DML Project

## License

This project is for educational purposes.
