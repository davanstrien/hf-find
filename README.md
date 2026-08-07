# hf-find

An [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) extension for **semantic search over the Hugging Face Hub** — datasets and models — from your terminal. Search by meaning, not keywords; find similar repos; see what's trending.

It queries a Hub semantic-search backend (embeddings + summaries over ~536k dataset cards and ~636k model cards) and is built to the house **agent-ergonomics contract**: stdout is data (TSV for agents, like native `hf`; JSON via `--format json`), stderr is hints, and every error carries its remedy.

## Install

```bash
hf extensions install davanstrien/hf-find
hf find --help
```

Uninstall: `hf extensions remove find`.

## Usage

```
hf find datasets "medieval manuscripts"        # semantic dataset search
hf find models   "OCR" --min-params 100000000  # model search + param filter
hf find similar  <repo_id>                      # similar repos (auto-detects type)
hf find trending [datasets|models]              # what's trending
hf find status                                  # backend health + index freshness
```

Common options: `-k N` (results, 1–100 — the backend caps at 100; `trending` also accepts `-k`, with `-n`/`--limit` as aliases), `--sort-by likes|downloads`, `--min-likes`, `--min-downloads`; models also take `--min-params` / `--max-params`, which accept size suffixes (`2B`, `500M`, `1.5B`) or plain ints. `similar` takes `--type dataset|model` to skip auto-detection (otherwise it resolves the type in one `/lookup` call). Invalid input (empty query, `-k 0`, `-k 200`, an unparseable size) fails fast client-side with a one-line remedy — never a raw backend error.

**Metadata filters** (on `datasets`, `models`, and `similar`): `--task` (exact tag, e.g. `text-generation`), `--license` (e.g. `apache-2.0`), `--language` (e.g. `en` — datasets only in practice; model cards rarely set it), and `--modified-since` (ISO date, e.g. `2026-01-01`). All are exact-match and combine with the semantic query:

```bash
hf find models "OCR" --task image-text-to-text --license apache-2.0
hf find datasets "speech" --language en --modified-since 2026-01-01
```

### `hf find datasets "medieval manuscripts" -k 5`

```
ID                                            SIM LIKES DLOADS SUMMARY
------------------------------------------- ----- ----- ------ -------------------------
wjbmattingly/medieval-letters-htr-synthetic 81.1%     0     97 This dataset provides 8,…
Vault-of-History/Medival_Manuscripts        80.8%     0      0 This dataset contains me…
BSC-CSSH/AMSMB-line-transcription           78.2%     0     45 This dataset provides 3,…
grackle-in-a-HEB-parking-lot/possumm-local  77.5%     0    488 This dataset contains 39…
jwidmer/missiven-inference                  77.5%     0      0 This dataset contains 2,…
```

(The summary column is truncated to fit the terminal width; likes/downloads are compacted, e.g. `22.2k`.)

### `hf find models "OCR" --min-params 100000000 -k 3`

```
ID                      SIM LIKES DLOADS PARAMS SUMMARY
--------------------- ----- ----- ------ ------ ----------------------------------------
PleIAs/OCRerrcr       72.1%    15      3 434.0M OCRerrcr is a DeBERTa-v2-based token cl…
colaibu/DeepSeek-OCR  68.5%     0      0   3.3B DeepSeek-OCR is a multilingual vision-l…
MdMonsur/dots.ocr-1.5 68.4%     0      0   3.0B dots.ocr-1.5 is a 3B-parameter multimod…
```

### `hf find similar wjbmattingly/medieval-letters-htr-synthetic -k 3` (type auto-detected)

```
ID                                                 SIM LIKES DLOADS SUMMARY
------------------------------------------------ ----- ----- ------ ------------------------
dh-unibe/image-text_bullinger-autoren            87.3%     0    142 This dataset contains 8…
samaritan-ai/samaritan_hebrew_LightOnOcr         87.2%     0     34 This dataset contains 4…
FatimahEmadEldin/Gutenberg-Arabic-OCR-HTML-Pages 87.1%     1     89 This dataset contains 1…
```

## Agent mode

When a coding agent is driving (detected via `CLAUDECODE`, `CODEX_SANDBOX`, `AI_AGENT`, … — reused from `huggingface_hub`'s own agent detection), output switches automatically, mirroring native `hf`: **TSV rows on stdout**, hints on stderr. Force it with `--format agent`; force human output with `--format human`.

TSV columns are `id`, `similarity` (raw float), `likes`, `downloads`, [`param_count` for models,] and `summary` last (free text; tabs/newlines escaped, so it parses as a clean table):

```bash
$ AI_AGENT=1 hf find models "OCR" -k 2   # stdout (tab-separated):
model_id	similarity	likes	downloads	param_count	summary
PleIAs/OCRerrcr	0.7209532856941223	15	3	434014210	OCRerrcr is a DeBERTa-v2-based token classification model for English, French, and German …
Fondelton/Richardson	0.6932852864265442	0	0	0	This English OCR model is a fine-tune of DeepSeek-OCR …
```

For a structured payload instead, pass **`--format json`** — one object per invocation:

```bash
$ hf find datasets "medieval manuscripts" -k 2 --format json   # stdout:
{"query": "medieval manuscripts", "type": "datasets", "count": 2, "results": [{"dataset_id": "wjbmattingly/medieval-letters-htr-synthetic", "similarity": 0.8107073307037354, "summary": "This dataset provides 8,317 synthetic medieval Latin manuscript pages ...", "likes": 0, "downloads": 97}, {"dataset_id": "Vault-of-History/Medival_Manuscripts", "similarity": 0.8079..., "summary": "...", "likes": 0, "downloads": 0}]}
```

The JSON object carries `{query, type, count, results[]}`; each result has `similarity`, `summary`, `likes`, `downloads` (and `param_count` for models). `similar` swaps `query` for `repo_id`; `trending` and `status` return their own shapes (`status` is a single object, so it emits JSON in both agent and json modes).

Hints go to **stderr** in `human` and `agent` modes:

```
Hint: explore: hf find similar wjbmattingly/medieval-letters-htr-synthetic
Hint: view: hf datasets card wjbmattingly/medieval-letters-htr-synthetic
```

**`--format json` and `--format quiet` keep stderr empty**, so a combined `2>&1` capture still parses cleanly — no need to remember `2>/dev/null`. (The two chatty modes, `human` and `agent`, are the ones a person or an agent reads interactively; the two machine formats stay silent.)

### Chained-discovery hints

Every result set ends with ≤2 hints (stderr) that point at **existing commands** the caller can run next — never a pre-fetched extra API call:

- after `datasets`/`models` → `explore: hf find similar <top-id>` and `view: hf datasets card <id>`
- after `similar` → the text-search variant and the card view
- after `trending` → `explore: hf find similar <top-id>`

### Errors carry their remedy

A bad repo id or an unreachable backend prints a one-line cause on stderr **and** the next step:

```
$ hf find similar this/does-not-exist-xyz          # stderr, exit 1:
Error: 'this/does-not-exist-xyz' not found as a dataset or model in the index
Hint: check the id, or search for it: hf find datasets "<name>"
```

## Bulk / offline access

The CLI is built for interactive, per-query lookups. When you want the *whole* corpus — filtering or aggregating across the Hub, joining against your own data, or working offline — go straight to the public dataset the backend is built from, [`davanstrien/search-v3-embeddings`](https://huggingface.co/datasets/davanstrien/search-v3-embeddings): one-sentence summaries, 1024-d embeddings, and card metadata for ~536k datasets and ~636k models. After any search, the CLI prints this as a stderr hint.

DuckDB reads it directly over `hf://`, pruning to the columns you name (so metadata-only queries never touch the vectors):

```bash
duckdb -c "
  SELECT id, summary, license, downloads
  FROM 'hf://datasets/davanstrien/search-v3-embeddings/models/part-*.parquet'
  WHERE task = 'text-generation' AND low_confidence = false
  ORDER BY downloads DESC
  LIMIT 20
"
```

The embeddings are in there too (fp16, 1024-d, null for refusal rows) for building your own index or running `array_cosine_similarity` in DuckDB — see the dataset card for the schema and a find-similar example.

## Backend

Default: `https://davanstrien-hub-search-api.hf.space`. Override with `HF_FIND_URL`.

> The backend is a free-tier Space — it may cold-start (~60s on first hit). `hf find status` shows readiness and index freshness.

## Notes

- Depends on `huggingface_hub`'s CLI output sink (`huggingface_hub.cli._output`), a **private API**, for agent detection and stderr styling — same approach as `hf-ds` / `hf-jobsx`. Loosely pinned (`>=1.12`) for the POC; pin exactly if the sink API shifts.
- No test suite yet; verified by hand and by blind agent testing.
