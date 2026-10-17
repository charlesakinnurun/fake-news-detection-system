# Engineering report

This report records what was implemented, what was actually executed, what the
results were, and what remains incomplete. It is written from the final working
tree, not from a plan.

## Status

The classical machine-learning system is complete end to end: data ingestion and
validation, leakage-safe splitting, candidate training, calibration, held-out
evaluation, artifact persistence, API serving, Streamlit UI, Docker packaging,
CI and documentation. The transformer extension is deliberately not trained (see
[Deviations](#deviations-and-engineering-decisions)).

## Implemented components

| Area | Implementation | Entry point |
|---|---|---|
| Dataset acquisition | Explicit ISOT download with checksum, progress and local-import path | `scripts/download_data.py` |
| Ingestion | Configurable column/label mapping, encoding detection, source provenance | `src/fake_news_detection/data/ingestion.py` |
| Validation | Empty/short/over-long content, labels, duplicates, conflicting labels, class balance, artifacts | `src/fake_news_detection/data/validation.py` |
| Preprocessing | Unicode NFKC, whitespace/control-character handling, documented title/body combination | `src/fake_news_detection/data/preprocessing.py` |
| Leakage-safe splitting | Exact de-duplication, MinHash/LSH near-duplicate clusters, grouped stratified split, audit | `src/fake_news_detection/data/splitting.py` |
| Features | Word TF-IDF (1–2 grams), optional character n-grams and univariate selection | `src/fake_news_detection/features/tfidf.py` |
| Candidates | Dummy, Logistic Regression, Complement NB, Random Forest, calibrated Linear SVC | `src/fake_news_detection/models/baseline.py` |
| Training | Grouped CV, validation comparison, selection, calibration, threshold tuning | `scripts/train.py` |
| Evaluation | Test metrics, confusion matrices, ROC/PR, calibration, latency, slices, error examples | `scripts/evaluate.py` |
| Artifact registry | Checksummed joblib pipeline + JSON metadata | `src/fake_news_detection/models/registry.py` |
| Inference | Shared validation and prediction path | `src/fake_news_detection/models/predict.py` |
| API | Health, readiness, metadata, metrics, single/batch predict, size/rate limits, request IDs | `src/fake_news_detection/api/` |
| UI | Streamlit form, probabilities, optional attribution, limitations panel | `app/streamlit_app.py` |
| Observability | Structured JSON logs, in-process latency counters | `src/fake_news_detection/monitoring/logging.py` |
| Deployment | Multi-target Dockerfile, Compose stack, non-root user, health checks | `Dockerfile`, `docker-compose.yml` |
| CI | Ruff, mypy, pytest, compile and startup smoke test | `.github/workflows/ci.yml` |
| Documentation | README, model card, this report | `README.md`, `docs/` |

## Verification performed

All commands were run from the repository root on Windows with Python 3.14.3 and
the dependency versions recorded in `artifacts/model_metadata.json` (scikit-learn
1.9.1, pandas 3.0.5, FastAPI 0.141.1, Streamlit 1.64.0).

| Check | Command | Result |
|---|---|---|
| Full training | `python scripts/train.py` | Completed in about 22 minutes on CPU; 5 candidates, grouped 3-fold CV; selected `linear_svc_calibrated` |
| Test evaluation | `python scripts/evaluate.py --latency-samples 200 --error-examples 12` | Completed; report and metadata updated; test macro-F1 0.9949 |
| Tests | `python -m pytest --cov=fake_news_detection --cov-report=term-missing` | **238 passed**, 84% statement/branch coverage, 9.09 s |
| Lint | `python -m ruff check .` | All checks passed |
| Formatting | `python -m ruff format --check .` | 54 files already formatted |
| Types | `python -m mypy src` | No issues in 27 source files |
| API startup | Uvicorn on 127.0.0.1:8000 | `/health` 200, `/ready` 200, `/metadata` 200 |
| API prediction | Live `POST /predict` calls | REAL sample → REAL (0.9167), FAKE sample → FAKE (1.0), batch → 200, explain → signed features |
| API validation | Empty body, unknown field, empty batch | 422 as documented |
| UI startup | `streamlit run app/streamlit_app.py --server.headless true` | `/_stcore/health` returned `ok`, root returned HTTP 200 |
| Compose config | `docker compose config` | Valid configuration |
| Make targets | `make` | `make` is not installed in this environment; every target mirrors a Python command that was run directly |
| Docker build | `docker build --target api -t fake-news-detection-api:local .` | **Blocked**: the Docker daemon is not running (`npipe:////./pipe/dockerDesktopLinuxEngine`); the Dockerfile could not be built in this environment |

The API smoke test also confirmed that request bodies are absent from the
structured logs and that a request ID is echoed in both the response body and the
`X-Request-ID` header.

## Results

The final artifact is the calibrated Linear SVC (`linear_svc_calibrated`,
v1.0.0) with a validation-tuned threshold of 0.40.

| Metric | Test value |
|---|---|
| Accuracy | 0.9950 |
| Balanced accuracy | 0.9947 |
| Precision (`FAKE`) | 0.9973 |
| Recall (`FAKE`) | 0.9916 |
| F1 (`FAKE`) | 0.9944 |
| Macro F1 | 0.9949 |
| Weighted F1 | 0.9950 |
| ROC-AUC | 0.9995 |
| Average precision | 0.9991 |
| Brier score | 0.0040 |
| Log loss | 0.0195 |
| Expected calibration error | 0.0024 |
| p50 / p95 latency | 6.15 ms / 10.58 ms |
| Artifact size | 9.46 MiB |

Confusion matrix (rows = actual, columns = predicted): 2,585 `FAKE` correctly
classified, 22 `FAKE` called `REAL`, 7 `REAL` called `FAKE`, 3,169 `REAL`
correctly classified.

Validation comparison: Linear SVC 0.9960 macro-F1, Logistic Regression 0.9913,
Random Forest 0.9904, Complement NB 0.9685, Dummy 0.3545. The winner also had
the highest positive-class F1; Logistic Regression is faster and smaller and is
the better fallback if latency or artifact size becomes the binding constraint.

These values are in-distribution and inflated by the corpus artifacts documented
in the model card. They must not be presented as generalisation to new
publishers, topics or time periods.

## Deviations and engineering decisions

- **Dependency groups use `pyproject.toml` extras** (`api`, `app`, `train`,
  `dev`, `transformer`) instead of a `requirements/` directory. This keeps one
  source of truth for packaging and avoids duplicated, drifting dependency
  lists.
- **No notebook is committed.** The quality report, training report, evaluation
  report and figures cover the exploratory and diagnostic needs of this project;
  a notebook would duplicate code that is already tested and executable.
- **No transformer was trained.** A complete, evaluated classical pipeline was
  prioritised over an incomplete DistilBERT run on CPU-only hardware. The
  `transformer` extra and the same evaluation protocol are in place for a future
  experiment.
- **Model artifacts and the dataset are not committed.** They are reproducible
  from the scripts, are excluded by `.gitignore`, and the dataset is not
  redistributed for licensing reasons. Docker mounts the artifact read-only
  rather than baking it into the image.
- **No fabricated screenshots or benchmark numbers.** The README reports only
  values produced by the commands above. The UI is verified by startup and API
  integration checks rather than an invented screenshot.

## Operational notes

### Updating the model

1. Change configuration or code deliberately and record the reason.
2. Bump `model_version` in `configs/training.yaml`.
3. Run `python scripts/prepare_data.py` if the data or split logic changed.
4. Run `python scripts/train.py` and inspect `reports/training_report.md`.
5. Run `python scripts/evaluate.py` once and inspect the test report.
6. Verify `GET /metadata` reports the new version and metrics.
7. Keep the previous `model.joblib` and `model_metadata.json` for rollback.

### Rollback

1. Restore the previous artifact and metadata pair (they must match; the checksum
   in the metadata is verified on load).
2. Restart the API process or container.
3. Confirm `GET /ready` returns 200 and `GET /metadata` reports the expected
   version.

### Resource requirements

- Training runs on CPU; no GPU is required for the classical pipeline.
- Inference is a CPU-only scikit-learn pipeline with a small artifact and
  single-digit-millisecond to low-tens-of-milliseconds latency; see the measured
  values in the test report.
- The API is stateless apart from the loaded model and the in-process rate-limit
  counters, so it can be scaled horizontally behind an ingress. Rate limiting
  must then move to the ingress or a shared store.

## Known limitations

- Evaluation is in-distribution and random (grouped by duplicate cluster), not
  source-held-out or temporal.
- Dataset artifacts inflate accuracy; see the model card and quality report.
- The model card's limitations are part of the deliverable, not a disclaimer
  added after the fact.
- Rate limiting is per process.
- Request-size limiting trusts the declared `Content-Length`; a streaming
  reverse proxy should enforce the limit for hostile clients.
- There is no out-of-distribution detector or production drift monitor in
  v1.0.0.

## GitHub repository description

Suggested text for the repository's About section:

> Leakage-aware fake-news classification with TF-IDF and classical models,
> rigorous held-out evaluation, a FastAPI service, a Streamlit UI, Docker and CI.

## Reproduction

```bash
python -m pip install -e ".[dev,api,app,train]"
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py
python -m pytest
```

The seed is `42`, the split strategy is `grouped_stratified`, the positive class
is `FAKE`, and the dataset fingerprint and package versions are recorded in the
generated metadata.
