"""``hf semsearch`` — semantic search over the Hugging Face Hub.

Queries Daniel's Hub semantic-search backend and applies the house
agent-ergonomics contract: stdout is data, stderr is hints; humans get an
aligned table, agents get native-style TSV (JSON via --format json); every
error carries its remedy; hints point at existing commands the caller can run
next (never a pre-fetch).

Agent-mode detection, ``--format`` handling, and stderr styling are borrowed
from ``huggingface_hub``'s own CLI output sink so this reads exactly like the
rest of the ``hf`` CLI. (That sink is a private API — see README for the pin.)
"""

from __future__ import annotations

import json
import shutil
import sys
from typing import Annotated, Any, Optional

import typer

# Reuse huggingface_hub's own CLI output sink (a private API) so this reads like
# the rest of `hf`: agent-mode auto-detection, `--format`, and stderr styling.
from huggingface_hub.cli._output import OutputFormat, out
from huggingface_hub.utils import tabulate

from . import client
from .client import SearchError

app = typer.Typer(
    name="semsearch",
    help="Semantic search over the Hugging Face Hub (datasets & models).",
    no_args_is_help=True,
    add_completion=False,
)

FormatOpt = Annotated[
    OutputFormat,
    typer.Option(
        "--format",
        help="Output format. 'auto' picks agent/human by context; agents get one JSON object.",
    ),
]


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _id_key(kind: str) -> str:
    return "dataset_id" if kind == "datasets" else "model_id"


def _fmt_count(n: Optional[int]) -> str:
    """Human-friendly integer (12 400 -> '12.4k')."""
    if n is None:
        return "-"
    if n < 1000:
        return str(n)
    for div, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if n >= div:
            return f"{n / div:.1f}{suffix}"
    return str(n)


def _human_table(items: list[dict[str, Any]], kind: str) -> str:
    id_key = _id_key(kind)
    is_models = kind == "models"

    # Fixed (non-summary) columns, formatted up front.
    headers = ["ID", "SIM", "LIKES", "DLOADS"]
    if is_models:
        headers.append("PARAMS")
    headers.append("SUMMARY")

    fixed_rows: list[list[str]] = []
    summaries: list[str] = []
    for it in items:
        row = [
            str(it.get(id_key, "")),
            f"{it.get('similarity', 0.0) * 100:.1f}%",
            _fmt_count(it.get("likes")),
            _fmt_count(it.get("downloads")),
        ]
        if is_models:
            row.append(_fmt_count(it.get("param_count")))
        fixed_rows.append(row)
        summaries.append(" ".join((it.get("summary") or "").split()))

    # Budget the summary column against the terminal width.
    term_width = shutil.get_terminal_size((100, 20)).columns
    fixed_widths = [
        max(len(h), *(len(r[i]) for r in fixed_rows))
        for i, h in enumerate(headers[:-1])
    ]
    gaps = len(headers) - 1  # tabulate joins columns with single spaces
    summary_budget = max(24, term_width - sum(fixed_widths) - gaps)

    rows = [
        fixed + [_truncate(summary, summary_budget)]
        for fixed, summary in zip(fixed_rows, summaries)
    ]
    aligns = {"SIM": "right", "LIKES": "right", "DLOADS": "right", "PARAMS": "right"}
    return tabulate(rows, headers=headers, alignments=aligns)


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: width - 1].rstrip() + "…"


def _agent_headers(kind: str) -> list[str]:
    """Column order for agent TSV: id, similarity, counts, [params,] summary last.

    Summary goes last because it is free text; the raw API field names are used
    (matching the JSON payload) and floats/ints are emitted unrounded for agents.
    """
    headers = [_id_key(kind), "similarity", "likes", "downloads"]
    if kind == "models":
        headers.append("param_count")
    headers.append("summary")
    return headers


def _emit(items: list[dict[str, Any]], kind: str, **envelope: Any) -> None:
    """Write results to stdout in the resolved output mode.

    Mirrors native `hf`: agents get TSV rows (the default when auto-detected),
    `--format json` gets one structured object, humans get an aligned table.
    """
    match out.mode:
        case OutputFormat.human:
            if not items:
                print("No results found.")
            else:
                print(_human_table(items, kind))
        case OutputFormat.json:
            print(json.dumps({**envelope, "type": kind, "count": len(items), "results": items}))
        case OutputFormat.quiet:
            for it in items:
                print(it.get(_id_key(kind), ""))
        case _:  # agent (and any other non-human mode): native TSV, no truncation
            out.table(items, headers=_agent_headers(kind))


def _fail(err: SearchError) -> None:
    out.error(err.message)
    if err.remedy:
        out.hint(err.remedy)
    raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# chained-discovery hints (stderr; point at existing commands, never pre-fetch)
# --------------------------------------------------------------------------- #
def _card_cmd(kind: str, repo_id: str) -> str:
    return f"hf {'datasets' if kind == 'datasets' else 'models'} card {repo_id}"


def _search_hints(items: list[dict[str, Any]], kind: str) -> None:
    if not items:
        out.hint("no matches — try broader terms or drop --min-likes/--min-downloads filters")
        return
    top = items[0].get(_id_key(kind), "")
    out.hint(f"explore: hf semsearch similar {top}")
    out.hint(f"view: {_card_cmd(kind, top)}")


def _similar_hints(items: list[dict[str, Any]], kind: str) -> None:
    if not items:
        return
    top = items[0].get(_id_key(kind), "")
    noun = "datasets" if kind == "datasets" else "models"
    out.hint(f'search by text: hf semsearch {noun} "<terms>"')
    out.hint(f"view: {_card_cmd(kind, top)}")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command()
def datasets(
    query: Annotated[str, typer.Argument(help="Free-text query.")],
    k: Annotated[int, typer.Option("-k", "--k", help="Number of results.")] = 10,
    sort_by: Annotated[Optional[str], typer.Option("--sort-by", help="Re-rank by 'likes' or 'downloads'.")] = None,
    min_likes: Annotated[Optional[int], typer.Option("--min-likes")] = None,
    min_downloads: Annotated[Optional[int], typer.Option("--min-downloads")] = None,
    format: FormatOpt = OutputFormat.auto,
) -> None:
    """Semantic search over datasets."""
    out.set_mode(format)
    try:
        items = client.search(
            "datasets", query, k=k, sort_by=sort_by, min_likes=min_likes, min_downloads=min_downloads
        )
    except SearchError as e:
        _fail(e)
    _emit(items, "datasets", query=query)
    _search_hints(items, "datasets")


@app.command()
def models(
    query: Annotated[str, typer.Argument(help="Free-text query.")],
    k: Annotated[int, typer.Option("-k", "--k", help="Number of results.")] = 10,
    sort_by: Annotated[Optional[str], typer.Option("--sort-by", help="Re-rank by 'likes' or 'downloads'.")] = None,
    min_likes: Annotated[Optional[int], typer.Option("--min-likes")] = None,
    min_downloads: Annotated[Optional[int], typer.Option("--min-downloads")] = None,
    min_params: Annotated[Optional[int], typer.Option("--min-params", help="Minimum parameter count.")] = None,
    max_params: Annotated[Optional[int], typer.Option("--max-params", help="Maximum parameter count.")] = None,
    format: FormatOpt = OutputFormat.auto,
) -> None:
    """Semantic search over models."""
    out.set_mode(format)
    try:
        items = client.search(
            "models",
            query,
            k=k,
            sort_by=sort_by,
            min_likes=min_likes,
            min_downloads=min_downloads,
            min_param_count=min_params,
            max_param_count=max_params,
        )
    except SearchError as e:
        _fail(e)
    _emit(items, "models", query=query)
    _search_hints(items, "models")


@app.command()
def similar(
    repo_id: Annotated[str, typer.Argument(help="Dataset or model id, e.g. 'org/name'.")],
    type: Annotated[Optional[str], typer.Option("--type", help="'dataset' or 'model'. Default: auto-detect.")] = None,
    k: Annotated[int, typer.Option("-k", "--k", help="Number of results.")] = 10,
    format: FormatOpt = OutputFormat.auto,
) -> None:
    """Find datasets or models similar to a given repo (auto-detects the type)."""
    out.set_mode(format)

    order = {"dataset": ["datasets"], "model": ["models"]}.get((type or "").lower(), ["datasets", "models"])
    last: SearchError | None = None
    for kind in order:
        try:
            items = client.similarity(kind, repo_id, k=k)
            _emit(items, kind, repo_id=repo_id)
            _similar_hints(items, kind)
            return
        except SearchError as e:
            last = e
            if not e.not_found:
                _fail(e)  # connectivity/5xx — don't mask it as "wrong type"
            # else: a clean 404, try the next kind
    _fail(
        SearchError(
            f"'{repo_id}' not found as a {' or '.join(t.rstrip('s') for t in order)} in the index",
            'check the id, or search for it: hf semsearch datasets "<name>"',
        )
        if last is None or last.not_found
        else last
    )


@app.command()
def trending(
    kind: Annotated[str, typer.Argument(help="'datasets' or 'models'.")] = "datasets",
    limit: Annotated[int, typer.Option("-n", "--limit", help="Number of results.")] = 10,
    format: FormatOpt = OutputFormat.auto,
) -> None:
    """Show trending datasets or models."""
    out.set_mode(format)
    if kind not in ("datasets", "models"):
        _fail(SearchError(f"unknown kind '{kind}'", "use 'datasets' or 'models': hf semsearch trending datasets"))
    try:
        items = client.trending(kind, limit=limit)
    except SearchError as e:
        _fail(e)
    _emit(items, kind)
    if items:
        out.hint(f"explore: hf semsearch similar {items[0].get(_id_key(kind), '')}")


@app.command()
def status(format: FormatOpt = OutputFormat.auto) -> None:
    """Check backend health and index freshness (wraps /health)."""
    out.set_mode(format)
    try:
        info = client.health()
    except SearchError as e:
        _fail(e)

    info["backend_url"] = client.base_url()
    match out.mode:
        case OutputFormat.human:
            counts = info.get("counts", {})
            ready = "ready" if info.get("ready") else "NOT ready"
            print(f"backend: {client.base_url()}  [{ready}]")
            print(f"  datasets indexed: {_fmt_count(counts.get('dataset_cards'))}")
            print(f"  models indexed:   {_fmt_count(counts.get('model_cards'))}")
            print(f"  embedding model:  {info.get('embedding_model')} (dim {info.get('dim')})")
            print(f"  index revision:   {info.get('index_revision')} @ {info.get('index_last_modified')}")
        case OutputFormat.quiet:
            print("ready" if info.get("ready") else "not-ready")
        case _:
            print(json.dumps(info))
    if not info.get("ready"):
        out.hint("index not ready — the Space may be booting; retry in ~60s")


def main() -> None:
    code = 0
    try:
        app()
    except SystemExit as exc:  # typer/click signal exit via SystemExit
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    sys.exit(code)


if __name__ == "__main__":
    main()
