# hf-semsearch

An [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) extension for **semantic search over the Hugging Face Hub** — datasets and models — from your terminal. Search by meaning, not keywords; find similar repos; see what's trending.

> **Naming is open.** `semsearch` is a working name. Alternatives: `hf sem`, `hf find`, `hf discover`. Change the `short_name` in `install_local.sh` and the `[project.scripts]` entry in `pyproject.toml` together to rename.

It queries a Hub semantic-search backend (embeddings + summaries over ~536k dataset cards and ~636k model cards) and is built to the house **agent-ergonomics contract**: stdout is data (TSV for agents, like native `hf`; JSON via `--format json`), stderr is hints, and every error carries its remedy.

## Install (local, no GitHub)

`hf extensions install` only fetches from public GitHub repos. For local development this repo ships `install_local.sh`, which replicates what the extension loader does — an isolated venv under the hf extensions root plus a `manifest.json` — so `hf semsearch` dispatches to it:

```bash
bash install_local.sh      # re-run any time to pick up code edits
hf semsearch --help
```

Uninstall: `hf extensions remove semsearch` (or `rm -rf ~/.local/share/hf/extensions/hf-semsearch`).

## Usage

```
hf semsearch datasets "medieval manuscripts"        # semantic dataset search
hf semsearch models   "OCR" --min-params 100000000  # model search + param filter
hf semsearch similar  <repo_id>                      # similar repos (auto-detects type)
hf semsearch trending [datasets|models]              # what's trending
hf semsearch status                                  # backend health + index freshness
```

Common options: `-k N` (results), `--sort-by likes|downloads`, `--min-likes`, `--min-downloads`; models also take `--min-params` / `--max-params`. `similar` takes `--type dataset|model` to skip auto-detection.

### `hf semsearch datasets "medieval manuscripts" -k 5`

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

### `hf semsearch models "OCR" --min-params 100000000 -k 3`

```
ID                      SIM LIKES DLOADS PARAMS SUMMARY
--------------------- ----- ----- ------ ------ ----------------------------------------
PleIAs/OCRerrcr       72.1%    15      3 434.0M OCRerrcr is a DeBERTa-v2-based token cl…
colaibu/DeepSeek-OCR  68.5%     0      0   3.3B DeepSeek-OCR is a multilingual vision-l…
MdMonsur/dots.ocr-1.5 68.4%     0      0   3.0B dots.ocr-1.5 is a 3B-parameter multimod…
```

### `hf semsearch similar wjbmattingly/medieval-letters-htr-synthetic -k 3` (type auto-detected)

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
$ AI_AGENT=1 hf semsearch models "OCR" -k 2   # stdout (tab-separated):
model_id	similarity	likes	downloads	param_count	summary
PleIAs/OCRerrcr	0.7209532856941223	15	3	434014210	OCRerrcr is a DeBERTa-v2-based token classification model for English, French, and German …
Fondelton/Richardson	0.6932852864265442	0	0	0	This English OCR model is a fine-tune of DeepSeek-OCR …
```

For a structured payload instead, pass **`--format json`** — one object per invocation:

```bash
$ hf semsearch datasets "medieval manuscripts" -k 2 --format json   # stdout:
{"query": "medieval manuscripts", "type": "datasets", "count": 2, "results": [{"dataset_id": "wjbmattingly/medieval-letters-htr-synthetic", "similarity": 0.8107073307037354, "summary": "This dataset provides 8,317 synthetic medieval Latin manuscript pages ...", "likes": 0, "downloads": 97}, {"dataset_id": "Vault-of-History/Medival_Manuscripts", "similarity": 0.8079..., "summary": "...", "likes": 0, "downloads": 0}]}
```

The JSON object carries `{query, type, count, results[]}`; each result has `similarity`, `summary`, `likes`, `downloads` (and `param_count` for models). `similar` swaps `query` for `repo_id`; `trending` and `status` return their own shapes (`status` is a single object, so it emits JSON in both agent and json modes).

Hints and errors go to **stderr**, so piping stdout to a TSV or JSON parser stays clean:

```
Hint: explore: hf semsearch similar wjbmattingly/medieval-letters-htr-synthetic
Hint: view: hf datasets card wjbmattingly/medieval-letters-htr-synthetic
```

### Chained-discovery hints

Every result set ends with ≤2 hints (stderr) that point at **existing commands** the caller can run next — never a pre-fetched extra API call:

- after `datasets`/`models` → `explore: hf semsearch similar <top-id>` and `view: hf datasets card <id>`
- after `similar` → the text-search variant and the card view
- after `trending` → `explore: hf semsearch similar <top-id>`

### Errors carry their remedy

A bad repo id or an unreachable backend prints a one-line cause on stderr **and** the next step:

```
$ hf semsearch similar this/does-not-exist-xyz          # stderr, exit 1:
Error: 'this/does-not-exist-xyz' not found as a dataset or model in the index
Hint: check the id, or search for it: hf semsearch datasets "<name>"
```

## Backend

Default: `https://davanstrien-hub-search-v3-dev.hf.space`. Override with `HF_SEMSEARCH_URL`.

> This default is a **`-dev` Space** — free-tier, may cold-start (~60s on first hit), and not a stability guarantee. Point `HF_SEMSEARCH_URL` at a production endpoint when one exists.

## Notes

- Depends on `huggingface_hub`'s CLI output sink (`huggingface_hub.cli._output`), a **private API**, for agent detection and stderr styling — same approach as `hf-ds` / `hf-jobsx`. Loosely pinned (`>=1.12`) for the POC; pin exactly if the sink API shifts.
- POC status: local only, no remote, no tests yet.
