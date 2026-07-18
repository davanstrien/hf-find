"""Thin HTTP client for the Hub semantic-search backend.

One class, one job: turn a semantic-search API call into either a list of
result dicts or a clean ``SearchError`` that already carries its own remedy.
No formatting, no CLI concerns — that all lives in ``cli.py``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_URL = "https://davanstrien-hub-search-v3-dev.hf.space"


def base_url() -> str:
    """Backend base URL, overridable via ``HF_SEMSEARCH_URL``."""
    return os.environ.get("HF_SEMSEARCH_URL", DEFAULT_URL).rstrip("/")


class SearchError(Exception):
    """An API failure that already knows how the caller should recover.

    ``remedy`` is a one-line next step (often an existing command); the CLI
    prints it to stderr so every error is actionable.
    """

    def __init__(self, message: str, remedy: str | None = None, *, not_found: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy
        self.not_found = not_found


# Generous read timeout: the -dev Space is free-tier and can cold-start (~60s).
_TIMEOUT = httpx.Timeout(90.0, connect=10.0)


def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{base_url()}{path}"
    clean = {k: v for k, v in params.items() if v is not None}
    try:
        resp = httpx.get(url, params=clean, timeout=_TIMEOUT)
    except httpx.TimeoutException as exc:
        raise SearchError(
            f"backend timed out after 90s ({base_url()})",
            "the -dev Space is free-tier and may be cold-starting; retry once in ~60s, "
            "or check it: hf semsearch status",
        ) from exc
    except httpx.HTTPError as exc:
        raise SearchError(
            f"cannot reach backend at {base_url()} ({exc})",
            "check the URL (HF_SEMSEARCH_URL) or the Space status: hf semsearch status",
        ) from exc

    if resp.status_code == 404:
        detail = _detail(resp)
        if detail is not None:  # our API's structured 404: a real "not in index"
            raise SearchError(detail, None, not_found=True)
        # HTML 404 from the host/gateway: wrong URL or the Space is gone.
        raise SearchError(
            f"backend returned 404 with no API body at {base_url()} — wrong URL or the Space is down?",
            "check HF_SEMSEARCH_URL, or the Space status: hf semsearch status",
        )
    if resp.status_code == 422:
        # Validation error: the request was malformed, so retrying won't help.
        # Client-side checks catch the common cases (k, params) before we get
        # here; this is the backstop, and it never says "retry".
        raise SearchError(
            "invalid request — the backend rejected an argument",
            "check the argument values (e.g. -k >= 1, valid --sort-by); this is not a transient error",
        )
    if resp.status_code >= 400:
        raise SearchError(
            f"backend returned HTTP {resp.status_code}: {_detail(resp) or resp.text[:200]}",
            "retry; if it persists, check the Space: hf semsearch status",
        )
    return resp.json()


def _detail(resp: httpx.Response) -> str | None:
    try:
        return resp.json().get("detail")
    except Exception:
        return None


def search(
    kind: str,
    query: str,
    *,
    k: int = 10,
    sort_by: str | None = None,
    min_likes: int | None = None,
    min_downloads: int | None = None,
    min_param_count: int | None = None,
    max_param_count: int | None = None,
    task: str | None = None,
    license: str | None = None,
    language: str | None = None,
    modified_after: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over ``datasets`` or ``models``."""
    params: dict[str, Any] = {
        "query": query,
        "k": k,
        "sort_by": sort_by,
        "min_likes": min_likes,
        "min_downloads": min_downloads,
        "task": task,
        "license": license,
        "language": language,
        "modified_after": modified_after,
    }
    if kind == "models":
        params["min_param_count"] = min_param_count
        params["max_param_count"] = max_param_count
    return _get(f"/search/{kind}", params)["results"]


def similarity(
    kind: str,
    repo_id: str,
    *,
    k: int = 10,
    task: str | None = None,
    license: str | None = None,
    language: str | None = None,
    modified_after: str | None = None,
) -> list[dict[str, Any]]:
    """Find repos similar to ``repo_id`` (raises ``SearchError`` on 404)."""
    key = "dataset_id" if kind == "datasets" else "model_id"
    params: dict[str, Any] = {
        key: repo_id,
        "k": k,
        "task": task,
        "license": license,
        "language": language,
        "modified_after": modified_after,
    }
    return _get(f"/similarity/{kind}", params)["results"]


def lookup(repo_id: str) -> str | None:
    """Resolve a repo id to its type in one call via ``/lookup/{repo_id}``.

    Returns ``"dataset"`` or ``"model"``. Returns ``None`` when the backend has
    no ``/lookup`` route (an older deployment) so the caller can fall back to the
    try-both-kinds probe. Raises ``SearchError(not_found=True)`` when the route
    exists and reports the id is in neither index.
    """
    url = f"{base_url()}/lookup/{repo_id}"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT)
    except httpx.TimeoutException as exc:
        raise SearchError(
            f"backend timed out after 90s ({base_url()})",
            "the -dev Space is free-tier and may be cold-starting; retry once in ~60s, "
            "or check it: hf semsearch status",
        ) from exc
    except httpx.HTTPError as exc:
        raise SearchError(
            f"cannot reach backend at {base_url()} ({exc})",
            "check the URL (HF_SEMSEARCH_URL) or the Space status: hf semsearch status",
        ) from exc

    if resp.status_code == 200:
        return resp.json().get("type")
    if resp.status_code == 404:
        detail = _detail(resp)
        # Our route's structured "not in either index" — a real miss.
        if detail and "in the index" in detail:
            raise SearchError(detail, 'check the id, or search for it: hf semsearch datasets "<name>"',
                              not_found=True)
        # FastAPI's default "Not Found" or an HTML 404: the route isn't there
        # (older backend). Signal fallback rather than a hard error.
        return None
    # Any other status: reuse the shared error mapping for a consistent message.
    if resp.status_code >= 400:
        raise SearchError(
            f"backend returned HTTP {resp.status_code}: {_detail(resp) or resp.text[:200]}",
            "retry; if it persists, check the Space: hf semsearch status",
        )
    return None


def trending(kind: str, *, limit: int = 10) -> list[dict[str, Any]]:
    return _get(f"/trending/{kind}", {"limit": limit})["results"]


def health() -> dict[str, Any]:
    return _get("/health", {})
