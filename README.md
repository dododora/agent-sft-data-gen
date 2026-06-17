# agent-sft-data-gen

Generate **STITCH-S single-turn SFT data** in Traditional Chinese (zh-TW) by
rewriting real multi-turn agentic conversations with **Gemma 4** (via the Google
Gemini API).

Each real turn — a user question plus the assistant's verbatim tool calls and
results — is handed to Gemma 4, which authors a *speaking-first* trajectory: it
talks to the user (`<SAY>`) while privately reasoning (`[SOPR]…[EOPR]`) and
splicing in the original `<TOOL_CALL>` / `<TOOL_RESULT>` lines unchanged. The
tool calls/results come from the dataset and are never invented or altered; only
the spoken and reasoning chunks are synthesized.

Source dataset:
https://huggingface.co/datasets/voidful/gemma4-agent-sft
https://huggingface.co/datasets/voidful/agent-sft

## Setup

Requires Python 3.10+ and a Google Gemini API key.

```bash
pip install requests pyarrow

cp .env.example .env
# then put your key in .env:
# GOOGLE_API_KEY=your_key_here
```

`callGemma.py` reads `GOOGLE_API_KEY` from the environment, falling back to the
project `.env` file.

## Usage

Generate single-turn rows from a dataset file (`.jsonl` or `.parquet`):

```bash
# process the whole file
python genData.py --data path/to/shard-00000.parquet --out out/single_turn.jsonl

# just the first record (smoke test)
python genData.py --data path/to/small_data.jsonl --out out/single_turn.jsonl --limit 1
```

Options:

| flag | default | meaning |
| --- | --- | --- |
| `--data` | `gemma4-agent-sft/canonical/small_data.jsonl` | input records (`.jsonl` or `.parquet`) |
| `--out` | `out/single_turn.jsonl` | single-turn `.jsonl` output |
| `--model` | `gemma-4-31b-it` (or `$GEMMA_MODEL`) | Gemini API model id |
| `--limit` | all | only process the first N records |
| `--start` | `0` | skip the first START records |


## Output

One JSON object per line in the output `.jsonl`:

| field | description |
| --- | --- |
| `id` | `{record_id}::turn{n}` |
| `source` | original dataset source tag (e.g. `glaive`, `toucan`) |
| `user` | the user's question, translated to zh-TW |
| `msg` | the complete STITCH-S trajectory as a single string |
| `input` | the exact per-turn INPUT object handed to the model, kept for traceability |

A trajectory uses these markers (see [prompt.txt](prompt.txt) for the full spec):

- `<SAY>…</SAY>` — user-visible zh-TW speech (opening, bridges, final answer)
- `[SOPR]…[EOPR]` — private reasoning, English, telegraphic
- `<TOOL_CALL>…</TOOL_CALL>` / `<TOOL_RESULT>…</TOOL_RESULT>` — copied verbatim from the data
- `[EOR]` — end of all reasoning; appears once, right before the final `<SAY>`

## How it works

[genData.py](genData.py) runs the pipeline:

1. **Load** records from `.jsonl` or `.parquet`. Parquet stores `messages`/`tools`
   as JSON strings and keeps each tool result as a standalone `{"role": "tool"}`
   message; loading parses the strings and folds tool messages back into the
   preceding assistant so the rest of the pipeline sees one canonical shape.
2. **Split** each conversation into turns (one user message + the assistant work
   that answers it, including tool calls/results).
3. **Build** the per-turn INPUT object that [prompt.txt](prompt.txt) expects and ask
   Gemma 4 to return `{"user": <zh-TW question>, "msg": <trajectory>}`.
4. **Validate** lightly (warn-only — the model's trajectory is never rewritten) and
   **write** one single-turn row per turn, feeding each turn forward as zh-TW
   context for the next.

[callGemma.py](callGemma.py) is a thin Gemini API client: it loads the key, retries
transient 5xx/429 errors with backoff, handles Gemma 4's hidden *thinking* part
(returning only the answer text and growing the token budget on truncation), and
can constrain output to JSON.

## Layout

```
callGemma.py   # Gemma 4 / Gemini API client
genData.py     # turn-splitting + generation pipeline
prompt.txt     # the STITCH-S rewriter system prompt
.env.example   # GOOGLE_API_KEY template
out/           # generated single-turn data
```
