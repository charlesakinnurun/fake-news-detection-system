"""Integration tests for the FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fake_news_detection.api.main import create_app
from fake_news_detection.config import ServiceSettings, get_service_settings

pytestmark = pytest.mark.integration

SAMPLE = {
    "title": "Council approves new cycle lane budget",
    "text": "The council voted to approve a budget for new cycle lanes after a two hour debate.",
}


class TestServiceEndpoints:
    def test_root_banner(self, api_client: TestClient) -> None:
        payload = api_client.get("/").json()
        assert payload["service"]
        assert "/predict" in payload["endpoints"]

    def test_health_reports_the_model_is_loaded(self, api_client: TestClient) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["uptime_seconds"] >= 0

    def test_ready_returns_200_when_loaded(self, api_client: TestClient) -> None:
        response = api_client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"ready": True, "model_loaded": True, "detail": None}

    def test_metadata_exposes_model_information(self, api_client: TestClient) -> None:
        body = api_client.get("/metadata").json()
        assert body["model_version"] == "0.0.1-test"
        assert body["labels"] == ["FAKE", "REAL"]
        assert body["positive_label"] == "FAKE"
        assert body["attribution_supported"] is True

    def test_metadata_does_not_leak_filesystem_paths(self, api_client: TestClient) -> None:
        raw = api_client.get("/metadata").text
        assert "model.joblib" not in raw
        assert ":\\" not in raw and "/tmp/" not in raw

    def test_metrics_endpoint(self, api_client: TestClient) -> None:
        api_client.get("/predict", headers={"accept": "application/json"})
        body = api_client.get("/metrics").json()
        assert "endpoints" in body and "uptime_seconds" in body

    def test_openapi_schema_is_generated(self, api_client: TestClient) -> None:
        schema = api_client.get("/openapi.json").json()
        assert "/predict" in schema["paths"]
        assert "/ready" in schema["paths"]

    def test_unknown_route_returns_404(self, api_client: TestClient) -> None:
        assert api_client.get("/does-not-exist").status_code == 404


class TestPredictEndpoint:
    def test_successful_prediction(self, api_client: TestClient) -> None:
        response = api_client.post("/predict", json=SAMPLE)
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in {"FAKE", "REAL"}
        assert set(body["probabilities"]) == {"FAKE", "REAL"}
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
        assert body["model_version"] == "0.0.1-test"
        assert body["decision_threshold"] == 0.5
        assert "model classification" in body["disclaimer"]

    def test_headline_only(self, api_client: TestClient) -> None:
        response = api_client.post("/predict", json={"title": SAMPLE["title"]})
        assert response.status_code == 200

    def test_body_only(self, api_client: TestClient) -> None:
        response = api_client.post("/predict", json={"text": SAMPLE["text"]})
        assert response.status_code == 200

    def test_attribution_is_returned_when_requested(self, api_client: TestClient) -> None:
        body = api_client.post("/predict", json={**SAMPLE, "explain": True}).json()
        assert body["attribution"]["supported"] is True
        assert body["attribution"]["method"] == "tfidf_linear_attribution"
        assert body["attribution"]["note"]

    def test_attribution_is_absent_by_default(self, api_client: TestClient) -> None:
        assert api_client.post("/predict", json=SAMPLE).json()["attribution"] is None

    @pytest.mark.parametrize(
        "payload",
        [
            {"title": "", "text": ""},
            {"title": "   ", "text": "\n\t"},
            {"title": None, "text": None},
            {},
        ],
    )
    def test_empty_payloads_are_rejected(self, api_client: TestClient, payload: dict) -> None:
        assert api_client.post("/predict", json=payload).status_code == 422

    def test_unknown_field_is_rejected(self, api_client: TestClient) -> None:
        response = api_client.post("/predict", json={**SAMPLE, "headline": "typo"})
        assert response.status_code == 422
        assert any("headline" in str(item["loc"]) for item in response.json()["detail"])

    def test_oversized_title_is_rejected(self, api_client: TestClient) -> None:
        response = api_client.post("/predict", json={"title": "x" * 1000, "text": SAMPLE["text"]})
        assert response.status_code == 422
        assert "max_title_chars" in response.json()["detail"][0]["msg"]

    def test_oversized_body_is_rejected(self, api_client: TestClient) -> None:
        response = api_client.post("/predict", json={"title": "t", "text": "y" * 30000})
        assert response.status_code == 422

    def test_validation_errors_are_json_serialisable(self, api_client: TestClient) -> None:
        response = api_client.post("/predict", json={"title": "", "text": ""})
        assert isinstance(response.json()["detail"], list)

    def test_request_id_is_generated_and_echoed(self, api_client: TestClient) -> None:
        response = api_client.post("/predict", json=SAMPLE)
        assert response.headers["X-Request-ID"].startswith("req_")
        assert response.json()["request_id"] == response.headers["X-Request-ID"]

    def test_client_supplied_request_id_is_propagated(self, api_client: TestClient) -> None:
        response = api_client.post("/predict", json=SAMPLE, headers={"X-Request-ID": "trace-123"})
        assert response.headers["X-Request-ID"] == "trace-123"

    def test_prediction_is_deterministic(self, api_client: TestClient) -> None:
        first = api_client.post("/predict", json=SAMPLE).json()
        second = api_client.post("/predict", json=SAMPLE).json()
        assert first["probabilities"] == second["probabilities"]


class TestBatchEndpoint:
    def test_batch_returns_one_prediction_per_item(self, api_client: TestClient) -> None:
        response = api_client.post("/predict/batch", json={"items": [SAMPLE, SAMPLE, SAMPLE]})
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 3
        assert body["model_version"] == "0.0.1-test"

    def test_batch_order_is_preserved(self, api_client: TestClient) -> None:
        items = [
            {
                "title": "council approved the budget",
                "text": "officials said the bridge will reopen",
            },
            {"title": "shocking secret cure", "text": "big pharma hides this miracle"},
        ]
        body = api_client.post("/predict/batch", json={"items": items}).json()
        single_first = api_client.post("/predict", json=items[0]).json()
        assert body["predictions"][0]["probabilities"] == single_first["probabilities"]

    def test_empty_batch_is_rejected(self, api_client: TestClient) -> None:
        assert api_client.post("/predict/batch", json={"items": []}).status_code == 422

    def test_oversized_batch_is_rejected(self, api_client: TestClient) -> None:
        response = api_client.post("/predict/batch", json={"items": [SAMPLE] * 100})
        assert response.status_code == 422
        assert "max_batch_size" in str(response.json()["detail"])


class TestUnavailableModel:
    def test_ready_reports_503(self, missing_model_settings: ServiceSettings) -> None:
        app = create_app(missing_model_settings)
        with TestClient(app) as client:
            response = client.get("/ready")
            assert response.status_code == 503
            assert response.json()["ready"] is False
            assert response.json()["detail"]

    def test_health_still_reports_the_process_is_up(
        self, missing_model_settings: ServiceSettings
    ) -> None:
        app = create_app(missing_model_settings)
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["model_loaded"] is False

    def test_predict_returns_503(self, missing_model_settings: ServiceSettings) -> None:
        app = create_app(missing_model_settings)
        with TestClient(app) as client:
            response = client.post("/predict", json=SAMPLE)
            assert response.status_code == 503
            assert "hint" in response.json()

    def test_metadata_returns_503(self, missing_model_settings: ServiceSettings) -> None:
        app = create_app(missing_model_settings)
        with TestClient(app) as client:
            assert client.get("/metadata").status_code == 503


class TestSafeguards:
    def test_rate_limit_returns_429(
        self, service_settings: ServiceSettings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FND_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("FND_RATE_LIMIT_REQUESTS", "3")
        monkeypatch.setenv("FND_RATE_LIMIT_WINDOW_SECONDS", "60")
        get_service_settings.cache_clear()
        app = create_app(get_service_settings())
        with TestClient(app) as client:
            statuses = [client.post("/predict", json=SAMPLE).status_code for _ in range(5)]
            assert statuses[:3] == [200, 200, 200]
            assert statuses[3] == 429
            assert client.get("/health").status_code == 200  # probes are exempt
        get_service_settings.cache_clear()

    def test_request_size_limit_returns_413(
        self, service_settings: ServiceSettings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FND_MAX_REQUEST_BYTES", "256")
        get_service_settings.cache_clear()
        app = create_app(get_service_settings())
        with TestClient(app) as client:
            response = client.post("/predict", json={"title": "t", "text": "y" * 2000})
            assert response.status_code == 413
        get_service_settings.cache_clear()

    def test_request_bodies_are_not_logged_by_default(self, api_client: TestClient, caplog) -> None:
        import logging

        secret = "UNIQUEMARKERSTRING"
        with caplog.at_level(logging.INFO):
            api_client.post("/predict", json={"title": secret, "text": secret})
        assert secret not in caplog.text
