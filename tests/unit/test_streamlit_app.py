"""Tests for the Streamlit front end's non-UI helpers.

The widgets themselves are exercised manually; these tests cover the logic that
decides whether a request is sent and how API failures are surfaced, which is
where user-visible bugs tend to live.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("streamlit_app_under_test", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_validate_inputs_rejects_empty(app_module: ModuleType) -> None:
    problems = app_module.validate_inputs("   ", "\n")
    assert problems
    assert "headline" in problems[0].lower()


def test_validate_inputs_accepts_headline_only(app_module: ModuleType) -> None:
    assert app_module.validate_inputs("A real headline", "") == []


def test_validate_inputs_enforces_limits(app_module: ModuleType) -> None:
    problems = app_module.validate_inputs("x" * (app_module.MAX_TITLE_CHARS + 1), "")
    assert any("headline" in problem.lower() for problem in problems)


def test_request_prediction_posts_expected_payload(
    app_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(200, {"prediction": "REAL", "probabilities": {"REAL": 0.9}})

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    result = app_module.request_prediction("http://api:8000/", "Headline", "", explain=True)

    assert result["prediction"] == "REAL"
    assert captured["url"] == "http://api:8000/predict"
    assert captured["json"] == {"title": "Headline", "text": None, "explain": True}


def test_request_prediction_reports_unavailable_model(
    app_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        app_module.requests, "post", lambda *args, **kwargs: FakeResponse(503, {"detail": "no"})
    )
    with pytest.raises(app_module.ApiError, match="no model is loaded"):
        app_module.request_prediction("http://api:8000", "title", "body")


def test_request_prediction_flattens_validation_errors(
    app_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "detail": [
            {"loc": ["body", "title"], "msg": "too long", "type": "value_error"},
            {"loc": ["body", "text"], "msg": "too short", "type": "value_error"},
        ]
    }
    monkeypatch.setattr(
        app_module.requests, "post", lambda *args, **kwargs: FakeResponse(422, payload)
    )
    with pytest.raises(app_module.ApiError) as error:
        app_module.request_prediction("http://api:8000", "title", "body")
    assert "title: too long" in str(error.value)
    assert "text: too short" in str(error.value)


def test_request_prediction_handles_transport_failure(
    app_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise app_module.requests.ConnectionError("refused")

    monkeypatch.setattr(app_module.requests, "post", boom)
    with pytest.raises(app_module.ApiError, match="Could not reach"):
        app_module.request_prediction("http://api:8000", "title", "body")
