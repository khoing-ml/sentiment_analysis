## Sentiment Analysis API

FastAPI service for sentiment prediction using an MLflow-tracked scikit-learn pipeline.

### Features
- Logistic Regression + TF-IDF pipeline
- MLflow model logging (with signature & example)
- Dynamic model loading via Model Registry or latest run fallback
- Health and model info endpoints
- Dockerized deployment
- Pytest API tests
- GitHub Actions CI

### Endpoints
- `GET /` Basic welcome
- `GET /health` Model load health
- `GET /model/info` Model metadata (run id, version if available)
- `POST /predict` Body: `{ "review": "text" }` returns sentiment

### Local Training
```bash
python scripts/train.py
```
Produces `latest_run_id.txt` and logs model to `mlruns/`.

### Run API Locally
```bash
uvicorn main:app --reload
```

### Docker
Build:
```bash
docker build -t sentiment-api .
```
Run (mount `mlruns` to access artifacts):
```bash
docker run -p 8000:8000 -v $(pwd)/mlruns:/app/mlruns sentiment-api
```

### Tests
```bash
pytest -q
```

### CI
GitHub Actions workflow at `.github/workflows/ci.yml` runs tests on push/PR to main.

### Next Ideas
- Rate limiting / auth
- Batch prediction endpoint
- Model drift monitoring
- Automated model promotion workflow