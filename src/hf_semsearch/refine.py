"""Relevance-feedback sessions for ``hf semsearch refine``.

A session is a named, on-disk preference state: an optional seed query vector
plus yes/no votes (each with the voted repo's stored embedding). The ranking
vector is a Rocchio blend — seed and (mean-of-yes − 0.6·mean-of-no), with the
vote side's weight growing as votes accumulate — matching the Hub Feeds
prototype so both surfaces learn the same way.

Pure-python vector math on 256-d lists; no numpy dependency.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def sessions_dir() -> Path:
    root = os.environ.get("HF_SEMSEARCH_HOME", "~/.cache/hf-semsearch")
    d = Path(root).expanduser() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name))


def _path(name: str) -> Path:
    return sessions_dir() / f"{name}.json"


def load(name: str) -> dict[str, Any] | None:
    p = _path(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def save(session: dict[str, Any]) -> None:
    session["updated"] = int(time.time())
    _path(session["name"]).write_text(json.dumps(session))


def delete(name: str) -> bool:
    p = _path(name)
    if p.exists():
        p.unlink()
        return True
    return False


def list_all() -> list[dict[str, Any]]:
    out = []
    for p in sorted(sessions_dir().glob("*.json")):
        s = load(p.stem)
        if s:
            out.append(s)
    return out


def new(name: str, kind: str) -> dict[str, Any]:
    return {"name": name, "kind": kind, "query": None, "qvec": None,
            "votes": [], "created": int(time.time()), "updated": int(time.time())}


# ---------------------------------------------------------------------------
# vector math
# ---------------------------------------------------------------------------
def _norm(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _mean(vecs: list[list[float]]) -> list[float]:
    m = [0.0] * len(vecs[0])
    for v in vecs:
        for i, x in enumerate(v):
            m[i] += x
    return [x / len(vecs) for x in m]


def preference_vector(session: dict[str, Any]) -> list[float] | None:
    """Rocchio blend of the seed query and vote feedback (None = no signal)."""
    pos = [v["vec"] for v in session["votes"] if v["dir"] > 0 and v.get("vec")]
    neg = [v["vec"] for v in session["votes"] if v["dir"] < 0 and v.get("vec")]
    vote_vec = None
    if pos or neg:
        dim = len((pos or neg)[0])
        v = [0.0] * dim
        if pos:
            for i, x in enumerate(_mean(pos)):
                v[i] += x
        if neg:
            for i, x in enumerate(_mean(neg)):
                v[i] -= 0.6 * x
        vote_vec = _norm(v)
    qvec = session.get("qvec")
    if qvec and vote_vec:
        wv = min(0.75, 0.25 + 0.06 * len(session["votes"]))
        return _norm([(1 - wv) * q + wv * b for q, b in zip(qvec, vote_vec)])
    return vote_vec or qvec


def record_votes(session: dict[str, Any], ids: list[str], direction: int,
                 vecs: dict[str, list[float]]) -> list[str]:
    """Add/replace votes for ids whose embeddings resolved; return missing ids."""
    missing = []
    for rid in ids:
        vec = vecs.get(rid)
        if vec is None:
            missing.append(rid)
            continue
        session["votes"] = [v for v in session["votes"] if v["id"] != rid]
        session["votes"].append({"id": rid, "dir": direction,
                                 "vec": [round(x, 5) for x in vec],
                                 "ts": int(time.time())})
    return missing
