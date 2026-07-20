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

from . import client, refine as refine_store
from .client import SearchError

# Rich collapses single newlines, so each line is its own paragraph (blank-line
# separated) to survive help rendering.
_EPILOG = (
    "Examples:\n\n"
    '  hf semsearch datasets "digitized herbarium specimens" -k 5\n\n'
    '  hf semsearch models "OCR" --max-params 2B   (size filter accepts 2B / 500M)\n\n'
    "  hf semsearch similar biglam/hmd_newspapers   (auto-detects dataset vs model)\n\n"
    "  hf semsearch trending datasets --format json   (one JSON object for scripts)\n\n"
    "Output: a table in a terminal; TSV when an agent is detected; --format json for a "
    "structured object; --format quiet for bare ids. Data on stdout, hints on stderr."
)

app = typer.Typer(
    name="semsearch",
    help="Semantic search over the Hugging Face Hub (datasets & models).",
    epilog=_EPILOG,
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """Semantic search over the Hugging Face Hub (datasets & models)."""


FormatOpt = Annotated[
    OutputFormat,
    typer.Option(
        "--format",
        help=(
            "Output format. Default: human table in a terminal, TSV when an agent is detected. "
            "Use 'json' for one structured object, 'quiet' for bare ids. Data on stdout, hints on stderr."
        ),
    ),
]

# Metadata filters — exact-match on the card fields, shared across commands.
TaskOpt = Annotated[
    Optional[str],
    typer.Option("--task", help="Filter by task/pipeline tag (exact match, e.g. text-generation)."),
]
LicenseOpt = Annotated[
    Optional[str],
    typer.Option("--license", help="Filter by license (exact match, e.g. apache-2.0)."),
]
LanguageOpt = Annotated[
    Optional[str],
    typer.Option("--language", help="Filter by language tag (exact match, e.g. en). Datasets only in practice."),
]
ModifiedSinceOpt = Annotated[
    Optional[str],
    typer.Option("--modified-since", help="Only repos modified on/after this ISO date (e.g. 2026-01-01)."),
]

# The public corpus behind the backend — surfaced as a bulk/offline escape hatch.
_CORPUS_REPO = "davanstrien/search-v3-embeddings"


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


def _hint(message: str) -> None:
    """Emit a stderr hint only in human/agent modes.

    `json` and `quiet` are machine formats: keep stderr empty so a `2>&1` capture
    still parses. This matches `quiet`'s native behavior and overrides the sink,
    which otherwise keeps hints in `json` mode.
    """
    if out.mode in (OutputFormat.human, OutputFormat.agent):
        out.hint(message)


def _fail(err: SearchError) -> None:
    out.error(err.message)
    if err.remedy:
        _hint(err.remedy)
    raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# input validation (client-side; never surface the backend's raw 422)
# --------------------------------------------------------------------------- #
def _validate_query(query: str) -> None:
    if not query.strip():
        _fail(SearchError("query is empty", 'provide search text, e.g. hf semsearch datasets "OCR datasets"'))


def _validate_k(k: int, flag: str = "-k") -> None:
    if k < 1:
        _fail(SearchError(f"{flag} must be >= 1 (got {k})", f"pass a positive count, e.g. {flag} 10"))


_SIZE_FACTORS = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}


def _parse_count(value: Optional[str], flag: str) -> Optional[int]:
    """Parse an int with an optional K/M/B/T suffix: '2B' -> 2_000_000_000.

    Plain ints ('2000000000'), decimals with a suffix ('1.5B'), and grouping
    separators ('1_000', '1,000') all work. Invalid input fails cleanly.
    """
    if value is None:
        return None
    s = value.strip().replace("_", "").replace(",", "")
    mult = 1
    if s and s[-1].lower() in _SIZE_FACTORS:
        mult = _SIZE_FACTORS[s[-1].lower()]
        s = s[:-1]
    try:
        n = int(float(s) * mult)
    except ValueError:
        _fail(SearchError(f"{flag}: '{value}' is not a valid size", f"use an int or a K/M/B suffix, e.g. {flag} 2B"))
    if n < 0:
        _fail(SearchError(f"{flag} must be >= 0 (got {value})", f"use a non-negative size, e.g. {flag} 500M"))
    return n


# --------------------------------------------------------------------------- #
# chained-discovery hints (stderr; point at existing commands, never pre-fetch)
# --------------------------------------------------------------------------- #
def _card_cmd(kind: str, repo_id: str) -> str:
    return f"hf {'datasets' if kind == 'datasets' else 'models'} card {repo_id}"


def _escape_hatch_hint(kind: str) -> None:
    """Point at the public corpus for bulk/offline work the API isn't shaped for.

    Two stderr lines (human/agent only): the repo, plus a one-line DuckDB example
    over its parquet. Filtering/aggregating the whole Hub is a query, not N API calls.
    """
    cfg = "datasets" if kind == "datasets" else "models"
    _hint(f"bulk/offline? the full corpus is public: hf://datasets/{_CORPUS_REPO}")
    _hint(
        f"  e.g. duckdb \"SELECT id, summary FROM "
        f"'hf://datasets/{_CORPUS_REPO}/{cfg}/part-*.parquet' LIMIT 10\""
    )


def _search_hints(items: list[dict[str, Any]], kind: str) -> None:
    if not items:
        _hint("no matches — try broader terms or drop --min-likes/--min-downloads filters")
        _escape_hatch_hint(kind)
        return
    top = items[0].get(_id_key(kind), "")
    _hint(f"explore: hf semsearch similar {top}")
    _hint(f"view: {_card_cmd(kind, top)}")
    _escape_hatch_hint(kind)


def _similar_hints(items: list[dict[str, Any]], kind: str) -> None:
    if not items:
        return
    top = items[0].get(_id_key(kind), "")
    noun = "datasets" if kind == "datasets" else "models"
    _hint(f'search by text: hf semsearch {noun} "<terms>"')
    _hint(f"view: {_card_cmd(kind, top)}")


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
    task: TaskOpt = None,
    license: LicenseOpt = None,
    language: LanguageOpt = None,
    modified_since: ModifiedSinceOpt = None,
    hybrid: Annotated[bool, typer.Option("--hybrid", help="Also match exact terms in the full card text (BM25), fused with semantic ranking.")] = False,
    format: FormatOpt = OutputFormat.auto,
) -> None:
    """Semantic search over datasets."""
    out.set_mode(format)
    _validate_query(query)
    _validate_k(k)
    try:
        items = client.search(
            "datasets", query, k=k, sort_by=sort_by, min_likes=min_likes, min_downloads=min_downloads,
            task=task, license=license, language=language, modified_after=modified_since,
            hybrid=hybrid,
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
    min_params: Annotated[
        Optional[str], typer.Option("--min-params", help="Minimum params, e.g. 100M, 1.5B, or 2000000000.")
    ] = None,
    max_params: Annotated[
        Optional[str], typer.Option("--max-params", help="Maximum params, e.g. 2B, 500M, or 2000000000.")
    ] = None,
    task: TaskOpt = None,
    license: LicenseOpt = None,
    language: LanguageOpt = None,
    modified_since: ModifiedSinceOpt = None,
    hybrid: Annotated[bool, typer.Option("--hybrid", help="Also match exact terms in the full card text (BM25), fused with semantic ranking.")] = False,
    format: FormatOpt = OutputFormat.auto,
) -> None:
    """Semantic search over models."""
    out.set_mode(format)
    _validate_query(query)
    _validate_k(k)
    min_pc = _parse_count(min_params, "--min-params")
    max_pc = _parse_count(max_params, "--max-params")
    try:
        items = client.search(
            "models",
            query,
            k=k,
            sort_by=sort_by,
            min_likes=min_likes,
            min_downloads=min_downloads,
            min_param_count=min_pc,
            max_param_count=max_pc,
            task=task,
            license=license,
            language=language,
            modified_after=modified_since,
            hybrid=hybrid,
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
    task: TaskOpt = None,
    license: LicenseOpt = None,
    language: LanguageOpt = None,
    modified_since: ModifiedSinceOpt = None,
    format: FormatOpt = OutputFormat.auto,
) -> None:
    """Find datasets or models similar to a given repo (auto-detects the type)."""
    out.set_mode(format)
    _validate_k(k)

    if (type or "").lower() in ("dataset", "model"):
        order = {"dataset": ["datasets"], "model": ["models"]}[type.lower()]
    else:
        # One /lookup call resolves the type instead of probing both routes. If
        # the route is absent (older backend), fall back to the try-both probe.
        try:
            resolved = client.lookup(repo_id)
        except SearchError as e:
            _fail(e)  # genuine not-in-index (not_found) or connectivity — surface it
        order = {"dataset": ["datasets"], "model": ["models"]}.get(resolved, ["datasets", "models"])

    last: SearchError | None = None
    for kind in order:
        try:
            items = client.similarity(kind, repo_id, k=k, task=task, license=license,
                                      language=language, modified_after=modified_since)
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
    # -k is the unified count flag across commands; -n/--limit kept as compat aliases.
    limit: Annotated[int, typer.Option("-k", "--limit", "-n", help="Number of results.")] = 10,
    format: FormatOpt = OutputFormat.auto,
) -> None:
    """Show trending datasets or models."""
    out.set_mode(format)
    if kind not in ("datasets", "models"):
        _fail(SearchError(f"unknown kind '{kind}'", "use 'datasets' or 'models': hf semsearch trending datasets"))
    _validate_k(limit)
    try:
        items = client.trending(kind, limit=limit)
    except SearchError as e:
        _fail(e)
    _emit(items, kind)
    if items:
        _hint(f"explore: hf semsearch similar {items[0].get(_id_key(kind), '')}")


def _split_ids(values: Optional[list[str]]) -> list[str]:
    """Flatten repeated/comma-separated id options into a clean list."""
    out: list[str] = []
    for v in values or []:
        out.extend(x.strip() for x in v.split(",") if x.strip())
    return out


@app.command()
def refine(
    session: Annotated[str, typer.Argument(help="Session name (created on first use). State persists across calls.")],
    query: Annotated[Optional[str], typer.Option("--query", "-q", help="Set/replace the session's seed topic.")] = None,
    yes: Annotated[Optional[list[str]], typer.Option("--yes", help="Mark repo id(s) relevant (repeat or comma-separate).")] = None,
    no: Annotated[Optional[list[str]], typer.Option("--no", help="Mark repo id(s) irrelevant (repeat or comma-separate).")] = None,
    type: Annotated[str, typer.Option("--type", help="'datasets' or 'models' (fixed at session creation).")] = "datasets",
    k: Annotated[int, typer.Option("-k", "--k", help="Number of results.")] = 10,
    reset: Annotated[bool, typer.Option("--reset", help="Discard the session's votes and seed, then start over.")] = False,
    format: FormatOpt = OutputFormat.auto,
) -> None:
    """Iterative search: seed with --query, vote with --yes/--no, re-run to refine.

    Each call folds the votes into a preference vector (Rocchio relevance
    feedback over the card embeddings) and returns results you haven't voted
    on. Only VOTED ids are excluded — re-running with no new votes re-pulls
    the same ranking, not a next page. First call: seed with --query, or vote
    directly on ids you found via 'hf semsearch datasets/models'. Similarity
    scores are comparative — trust the rank order, not the absolute values.
    On multi-topic briefs, spread --yes votes across sub-topics: the ranking
    follows your votes, so piling onto one theme narrows it (counter-vote or
    re-seed with --query to re-broaden).
    """
    out.set_mode(format)
    _validate_k(k)
    if not refine_store.valid_name(session):
        _fail(SearchError(f"invalid session name '{session}'",
                          "use letters/digits/dash/underscore, e.g. hf semsearch refine my-task"))
    if type not in ("datasets", "models"):
        _fail(SearchError(f"unknown --type '{type}'", "use 'datasets' or 'models'"))

    sess = None if reset else refine_store.load(session)
    if sess is None:
        sess = refine_store.new(session, type)
    kind = sess["kind"]

    try:
        if query is not None:
            if not query.strip():
                _fail(SearchError("--query is empty", 'provide topic text, e.g. --query "historical newspapers OCR"'))
            sess["query"] = query
            sess["qvec"] = client.embed_text(query, kind)

        yes_ids, no_ids = _split_ids(yes), _split_ids(no)
        missing: list[str] = []
        if yes_ids or no_ids:
            vecs = client.embeddings_batch(kind, yes_ids + no_ids)
            missing += refine_store.record_votes(sess, yes_ids, 1, vecs)
            missing += refine_store.record_votes(sess, no_ids, -1, vecs)

        vec = refine_store.preference_vector(sess)
        if vec is None:
            _fail(SearchError(
                f"session '{session}' has no signal yet",
                f'seed it: hf semsearch refine {session} --query "<topic>"  (or vote: --yes org/name)',
            ))

        voted_ids = [v["id"] for v in sess["votes"]]
        items = client.search_vector(kind, vec, k=k, exclude_ids=voted_ids)
    except SearchError as e:
        _fail(e)

    refine_store.save(sess)
    id_key = _id_key(kind)
    for it in items:
        it[id_key] = it.pop("id", it.get(id_key, ""))
        it.pop("embedding", None)
    n_yes = sum(1 for v in sess["votes"] if v["dir"] > 0)
    n_no = len(sess["votes"]) - n_yes
    _emit(items, kind, session=session, query=sess.get("query"), votes_yes=n_yes, votes_no=n_no)

    for rid in missing:
        out.error(f"'{rid}' not in the index — vote skipped")
    if missing:
        _hint('check ids with: hf semsearch similar <id>  (new repos appear after the daily refresh)')
    if items:
        top = items[0].get(id_key, "")
        _hint(f"vote, then re-run to refine: hf semsearch refine {session} --yes {top} --no <id>")
        _hint(f"session state: {n_yes}✓ {n_no}✗ · list sessions: hf semsearch sessions")


@app.command()
def sessions(
    delete: Annotated[Optional[str], typer.Option("--delete", help="Delete the named session.")] = None,
    format: FormatOpt = OutputFormat.auto,
) -> None:
    """List (or delete) refine sessions."""
    out.set_mode(format)
    if delete is not None:
        if refine_store.delete(delete):
            print(f"deleted session '{delete}'")
        else:
            _fail(SearchError(f"no session named '{delete}'", "list sessions: hf semsearch sessions"))
        return
    rows = [
        {
            "session": s["name"],
            "type": s["kind"],
            "query": s.get("query") or "-",
            "yes": sum(1 for v in s["votes"] if v["dir"] > 0),
            "no": sum(1 for v in s["votes"] if v["dir"] < 0),
        }
        for s in refine_store.list_all()
    ]
    match out.mode:
        case OutputFormat.human:
            if not rows:
                print("No refine sessions yet.")
                _hint('start one: hf semsearch refine my-task --query "<topic>"')
            else:
                print(tabulate([list(r.values()) for r in rows],
                               headers=["SESSION", "TYPE", "QUERY", "YES", "NO"]))
        case OutputFormat.json:
            print(json.dumps({"count": len(rows), "sessions": rows}))
        case OutputFormat.quiet:
            for r in rows:
                print(r["session"])
        case _:
            out.table(rows, headers=["session", "type", "query", "yes", "no"])


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
        _hint("index not ready — the Space may be booting; retry in ~60s")


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
