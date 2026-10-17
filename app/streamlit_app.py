"""Streamlit front end for the fake-news classification API.

The app contains no model logic of its own: every prediction is delegated to
``POST /predict`` on the FastAPI service. That keeps a single inference path
and means the UI can never drift from the deployed model.

Run it with::

    streamlit run app/streamlit_app.py

Set ``FND_API_BASE_URL`` when the API is not on ``http://127.0.0.1:8000``.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 30
METADATA_TIMEOUT_SECONDS = 5

MAX_TITLE_CHARS = 512
MAX_TEXT_CHARS = 20_000
MIN_TOTAL_CHARS = 4

LIMITATIONS = """
**What this result means.** The model compares the text with statistical
patterns learned from a labelled research corpus. It is not a fact-check, it
does not retrieve sources, and it cannot establish whether a claim is true.
A high probability only means the text resembles one side of the training
data. Publisher style, topic and vocabulary drift can all change the result.
Use it to prioritise human review, never as the sole basis for a decision
about a person, publisher or organisation.
"""


class ApiError(RuntimeError):
    """A request to the prediction API failed in a way worth showing the user."""


def api_base_url() -> str:
    """Return the configured API base URL without a trailing slash."""
    return os.environ.get("FND_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def validate_inputs(title: str, text: str) -> list[str]:
    """Return human-readable validation problems for the current form values.

    The server enforces the same limits; checking here avoids a pointless round
    trip and gives a friendlier message.
    """
    problems: list[str] = []
    title = title or ""
    text = text or ""
    if len(title) > MAX_TITLE_CHARS:
        problems.append(f"The headline is longer than {MAX_TITLE_CHARS:,} characters.")
    if len(text) > MAX_TEXT_CHARS:
        problems.append(f"The article body is longer than {MAX_TEXT_CHARS:,} characters.")
    if not title.strip() and not text.strip():
        problems.append("Enter a headline, an article body, or both.")
    elif len(title.strip()) + len(text.strip()) < MIN_TOTAL_CHARS:
        problems.append("The combined input is too short to classify.")
    return problems


def _detail_message(detail: Any) -> str:
    """Flatten FastAPI's validation-error payloads into readable text."""
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                location = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
                message = str(item.get("msg", "invalid value"))
                parts.append(f"{location}: {message}" if location else message)
            else:
                parts.append(str(item))
        return "; ".join(parts) if parts else "The request was rejected."
    return str(detail)


def request_prediction(
    base_url: str,
    title: str,
    text: str,
    *,
    explain: bool = False,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Call ``POST /predict`` and return the decoded response.

    Raises :class:`ApiError` with a user-safe message for transport failures,
    rejected input and an unavailable model.
    """
    payload: dict[str, Any] = {"title": title or None, "text": text or None}
    if explain:
        payload["explain"] = True
    try:
        response = requests.post(f"{base_url.rstrip('/')}/predict", json=payload, timeout=timeout)
    except requests.RequestException as error:
        raise ApiError(
            f"Could not reach the prediction API at {base_url}. "
            "Check that the service is running and that the URL is correct."
        ) from error

    if response.status_code == 503:
        raise ApiError(
            "The API is running but no model is loaded. Train or mount the model "
            "artifacts, then check the service's /ready endpoint."
        )
    if response.status_code != 200:
        try:
            detail = _detail_message(response.json().get("detail", response.text))
        except ValueError:
            detail = response.text or f"HTTP {response.status_code}"
        raise ApiError(f"Prediction was rejected (HTTP {response.status_code}): {detail}")
    try:
        body = response.json()
    except ValueError as error:
        raise ApiError("The prediction API returned a response that was not valid JSON.") from error
    if not isinstance(body, dict) or "prediction" not in body:
        raise ApiError("The prediction API returned an unexpected response.")
    return body


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_metadata(base_url: str) -> dict[str, Any] | None:
    """Fetch ``GET /metadata`` once a minute; return ``None`` when unavailable."""
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/metadata", timeout=METADATA_TIMEOUT_SECONDS
        )
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _render_sidebar(base_url: str) -> tuple[str, dict[str, Any] | None]:
    """Render service status and allow the API URL to be overridden."""
    with st.sidebar:
        st.header("Service")
        base_url = st.text_input(
            "API base URL",
            value=base_url,
            help="The FastAPI service. Override with the FND_API_BASE_URL environment variable.",
        ).rstrip("/")
        metadata = _fetch_metadata(base_url) if base_url else None
        if metadata:
            st.success("Model loaded")
            st.caption(
                f"{metadata.get('model_name', 'model')} · v{metadata.get('model_version', '?')}"
            )
            threshold = metadata.get("decision_threshold")
            if isinstance(threshold, (int, float)):
                st.caption(f"Decision threshold: {threshold:.2f}")
        else:
            st.warning("Model metadata unavailable")
            st.caption("The API may still be starting, or no model artifact is loaded.")
        st.divider()
        st.markdown("**Endpoints**")
        st.code(f"{base_url}\n{base_url}/docs", language="text")
    return base_url, metadata


def _render_probabilities(probabilities: dict[str, Any]) -> None:
    """Show one accessible progress bar per class."""
    numeric = {
        str(label): float(value)
        for label, value in probabilities.items()
        if isinstance(value, (int, float))
    }
    for label, value in sorted(numeric.items(), key=lambda item: item[1], reverse=True):
        st.progress(min(max(value, 0.0), 1.0), text=f"{label}: {value:.1%}")


def _render_attribution(attribution: dict[str, Any] | None) -> None:
    """Render signed TF-IDF contributions when the model supports them."""
    if not attribution:
        return
    st.subheader("Feature attribution")
    if not attribution.get("supported"):
        st.info(attribution.get("reason") or "Attribution is not available for this model.")
        return
    features = attribution.get("features") or []
    if not features:
        st.info("No non-zero TF-IDF features were found for this input.")
        return
    frame = pd.DataFrame(features)
    frame = frame.rename(
        columns={"feature": "Feature", "contribution": "Contribution", "direction": "Towards"}
    )
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption(
        "Signed contributions from a linear TF-IDF model: positive values push toward "
        "the positive class, negative values away from it. This is a model explanation, "
        "not evidence about factual accuracy."
    )


def _render_result(result: dict[str, Any], metadata: dict[str, Any] | None) -> None:
    """Render a prediction response, keeping the caveats prominent."""
    prediction = str(result.get("prediction", "UNKNOWN"))
    probabilities = result.get("probabilities") or {}
    confidence = probabilities.get(prediction)
    st.subheader("Result")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Model classification", prediction)
    with right:
        if isinstance(confidence, (int, float)):
            st.metric("Probability for this class", f"{float(confidence):.1%}")
        threshold = result.get("decision_threshold")
        if isinstance(threshold, (int, float)):
            st.caption(f"Positive-class decision threshold: {float(threshold):.2f}")
    if isinstance(probabilities, dict) and probabilities:
        st.markdown("**Class probabilities**")
        _render_probabilities(probabilities)

    _render_attribution(result.get("attribution"))

    st.warning(LIMITATIONS.strip())
    model_version = result.get("model_version") or (metadata or {}).get("model_version", "?")
    model_name = result.get("model_name") or (metadata or {}).get("model_name", "model")
    st.caption(f"Served by {model_name} v{model_version}.")


def main() -> None:
    """Render the page."""
    st.set_page_config(
        page_title="Fake News Detection",
        page_icon="📰",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    base_url, metadata = _render_sidebar(api_base_url())

    st.title("Fake News Detection")
    st.caption(
        "A research classifier that estimates whether an article resembles the "
        "REAL or FAKE examples in a labelled corpus. It is not a fact-check."
    )

    with st.form("prediction_form"):
        title = st.text_input(
            "Headline",
            max_chars=MAX_TITLE_CHARS,
            placeholder="Example: Council approves annual budget after debate",
        )
        text = st.text_area(
            "Article body",
            height=240,
            max_chars=MAX_TEXT_CHARS,
            placeholder="Paste the article text here. Headline only is also accepted.",
        )
        explain_default = bool((metadata or {}).get("attribution_supported"))
        explain = st.checkbox(
            "Show influential features (linear models only)",
            value=False,
            disabled=not explain_default,
            help=(
                "Requests signed TF-IDF contributions from the API. "
                "Only meaningful for the linear candidate."
            ),
        )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        problems = validate_inputs(title, text)
        if problems:
            for problem in problems:
                st.error(problem)
        else:
            with st.spinner("Classifying…"):
                try:
                    result = request_prediction(base_url, title, text, explain=explain)
                except ApiError as error:
                    st.error(str(error))
                else:
                    st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if result:
        _render_result(result, metadata)

    with st.expander("How to read this result"):
        st.markdown(LIMITATIONS.strip())
        st.markdown(
            "The model card in `docs/MODEL_CARD.md` documents the training data, "
            "evaluation protocol and known limitations in more detail."
        )


if __name__ == "__main__":
    main()
