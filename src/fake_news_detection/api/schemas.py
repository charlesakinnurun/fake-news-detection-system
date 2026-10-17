"""Request and response schemas.

The length limits come from :class:`~fake_news_detection.config.ServiceSettings`
so that the published OpenAPI document reflects how the deployed instance is
actually configured.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fake_news_detection.config import get_service_settings

__all__ = [
    "AttributionResponse",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "HealthResponse",
    "MetadataResponse",
    "PredictRequest",
    "PredictionResponse",
    "ReadyResponse",
]


class StrictRequest(BaseModel):
    """Base request model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class PredictRequest(StrictRequest):
    """A single article to classify."""

    title: str | None = Field(
        default=None,
        description="Headline. Either `title` or `text` (or both) must be provided.",
        examples=["Example news headline"],
    )
    text: str | None = Field(
        default=None,
        description="Full article body. Either `title` or `text` (or both) must be provided.",
        examples=["The complete news article text goes here."],
    )
    explain: bool | None = Field(
        default=None,
        description=(
            "Request signed TF-IDF feature attribution. Only available for linear models; "
            "the response reports `supported: false` otherwise."
        ),
    )

    @model_validator(mode="after")
    def _validate_content(self) -> PredictRequest:
        settings = get_service_settings()
        title = self.title or ""
        text = self.text or ""

        if len(title) > settings.max_title_chars:
            raise ValueError(
                f"title exceeds max_title_chars={settings.max_title_chars} (got {len(title)})"
            )
        if len(text) > settings.max_text_chars:
            raise ValueError(
                f"text exceeds max_text_chars={settings.max_text_chars} (got {len(text)})"
            )
        if not title.strip() and not text.strip():
            raise ValueError("provide a non-empty 'title', 'text', or both")
        if len(title.strip()) + len(text.strip()) < settings.min_total_chars:
            raise ValueError(
                f"combined input is shorter than min_total_chars={settings.min_total_chars}"
            )
        return self


class BatchPredictRequest(StrictRequest):
    """A batch of articles to classify in one call."""

    items: list[PredictRequest] = Field(min_length=1, description="Articles to classify.")

    @model_validator(mode="after")
    def _validate_batch_size(self) -> BatchPredictRequest:
        settings = get_service_settings()
        if len(self.items) > settings.max_batch_size:
            raise ValueError(
                f"batch size {len(self.items)} exceeds max_batch_size={settings.max_batch_size}"
            )
        return self


class AttributionResponse(BaseModel):
    """Signed TF-IDF feature attribution for a single prediction."""

    supported: bool
    method: str | None = None
    reason: str | None = None
    features: list[dict[str, Any]] = Field(default_factory=list)
    note: str = ""


class PredictionResponse(BaseModel):
    """The classification result for one article."""

    prediction: str = Field(description="Predicted label at the decision threshold.")
    probabilities: dict[str, float] = Field(
        description="Per-class probability from the model's `predict_proba`."
    )
    score: float = Field(description="Probability assigned to the positive class.")
    positive_label: str = Field(description="The class the binary metrics refer to.")
    decision_threshold: float
    model_version: str
    model_name: str
    request_id: str | None = None
    disclaimer: str
    attribution: AttributionResponse | None = None


class BatchPredictResponse(BaseModel):
    """Predictions for a batch request, in input order."""

    predictions: list[PredictionResponse]
    model_version: str
    disclaimer: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Liveness information. Contains no secrets or filesystem paths."""

    status: str
    version: str
    service: str
    uptime_seconds: float
    model_loaded: bool


class ReadyResponse(BaseModel):
    """Readiness information for orchestrators."""

    ready: bool
    model_loaded: bool
    detail: str | None = None


class MetadataResponse(BaseModel):
    """Safe, non-sensitive model metadata."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    created_at: str
    labels: list[str]
    positive_label: str
    decision_threshold: float
    text_field: str
    calibration: dict[str, Any] | None = None
    training: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] | None = None
    attribution_supported: bool = False
    disclaimer: str = ""
