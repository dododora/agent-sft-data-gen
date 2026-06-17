"""Generate STITCH-S single-turn training data from the gemma4-agent-sft canon.

The LLM writes the WHOLE trajectory itself. This script only prepares the input
(splitting the multi-turn conversation into turns and handing the model the real,
verbatim tool calls/results) and then stores what the model returns. There is no
code-side template that assembles the trajectory — the model authors every
<SAY> / [SOPR] chunk and decides the structure, copying the <TOOL_CALL> /
<TOOL_RESULT> lines verbatim from the input.

Pipeline
--------
1. Read the multi-turn records from a .jsonl OR .parquet file (see load_records).
   The raw agent-sft parquet stores `messages`/`tools` as JSON strings and keeps
   each tool result in a standalone {"role": "tool"} message; load_records parses
   the strings and folds those tool messages back into the preceding assistant, so
   the rest of the pipeline always sees the canonical embedded shape.
2. Split each conversation into individual turns (one user message + the
   assistant work that answers it, including any tool calls/results).
3. For every turn build the per-turn INPUT object that prompt.txt expects and ask
   Gemma 4 to return {"user": <zh-TW question>, "msg": <full STITCH-S trajectory>}.
4. Flatten the multi-turn conversation into single-turn rows and save them to
   gemma4-agent-sft/out/ with fields: id, source, user, msg.

Run (needs GOOGLE_API_KEY in .env):
    python gemma4-agent-sft/genData.py            # process the whole file
    python gemma4-agent-sft/genData.py --limit 1  # just the first record
    python gemma4-agent-sft/genData.py --data path/to/shard-00000.parquet
"""

from __future__ import annotations

import os
import re
import sys
import json
import argparse
import traceback

# callGemma.py lives at the repo root (one level up from this file).
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from callGemma import call_gemma, _extract_json  # noqa: E402

DATA_PATH = os.path.join(HERE, "gemma4-agent-sft", "canonical", "small_data.jsonl")
PROMPT_PATH = os.path.join(HERE, "prompt.txt")
OUT_DIR = os.path.join(HERE, "out")
OUT_PATH = os.path.join(OUT_DIR, "single_turn.jsonl")


# --------------------------------------------------------------------------- #
# Record loading (.jsonl or .parquet) + format normalization
# --------------------------------------------------------------------------- #
def _collapse_tool_messages(messages: list[dict]) -> list[dict]:
    """Fold standalone {"role": "tool"} messages into the preceding assistant.

    The raw agent-sft data stores a tool result as its own message
    ({"role": "tool", "tool_responses": [...]}) right after the assistant message
    that holds the matching `tool_calls`. The canonical jsonl (and the rest of this
    pipeline) instead expects `tool_responses` embedded in that assistant message.
    This rewrites the former into the latter; already-collapsed input is unchanged
    (there are no tool-role messages to fold).
    """
    out: list[dict] = []
    last_assistant: dict | None = None
    for msg in messages:
        if msg.get("role") == "tool":
            if last_assistant is not None:
                last_assistant.setdefault("tool_responses", []).extend(
                    msg.get("tool_responses") or []
                )
            continue  # drop the standalone tool message
        msg = dict(msg)  # copy so we never mutate the source record
        out.append(msg)
        if msg.get("role") == "assistant":
            last_assistant = msg
    return out


def _parse_record(rec: dict) -> dict:
    """Normalize one raw record into the shape the pipeline expects.

    Parquet stores `messages`/`tools` as JSON strings; parse them if needed, then
    collapse standalone tool messages into their assistant.
    """
    rec = dict(rec)
    msgs = rec.get("messages")
    if isinstance(msgs, str):
        msgs = json.loads(msgs) if msgs else []
    tools = rec.get("tools")
    if isinstance(tools, str):
        tools = json.loads(tools) if tools else []
    rec["messages"] = _collapse_tool_messages(msgs or [])
    rec["tools"] = tools or []
    return rec


def load_records(path: str) -> list[dict]:
    """Load records from a .jsonl or .parquet file, normalized for the pipeline."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq  # lazy: only needed for parquet input

        table = pq.read_table(path, columns=["id", "source", "messages", "tools"])
        return [_parse_record(row) for row in table.to_pylist()]

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_parse_record(json.loads(line)))
    return records


# --------------------------------------------------------------------------- #
# System prompt: reuse prompt.txt's rules and ask for the FULL trajectory in one
# {user, msg} object. The model authors the whole `msg`.
# --------------------------------------------------------------------------- #
# The complete worked example (exactly the trajectory the user wants the model to
# emit). Built as a real string, then JSON-encoded so the model sees valid,
# properly escaped JSON in the prompt.
_EXAMPLE_MSG = """<SAY>沒問題，要精準地記錄馬拉松的官方起跑時刻，時間當然一定要抓得非常標準。我馬上幫您查詢目前系統的即時標準時間，確保資訊是正確的。</SAY>

[SOPR]Task: get exact current datetime for marathon official start. Next: call time_mcp_server_current_time with format YYYY-MM-DD HH:mm:ss.[EOPR]

<TOOL_CALL>{"function": {"name": "time_mcp_server_current_time", "arguments": {"format": "YYYY-MM-DD HH:mm:ss"}}}</TOOL_CALL>

<SAY>我正在向時間服務查詢目前標準的時間，會用到秒的精準度。請您稍微等一下，拿到之後就直接給您可以記錄的完整時刻。</SAY>

<TOOL_RESULT>{"name": "time_mcp_server_current_time", "response": "Current UTC time is 2025-08-27 23:23:29, and the time in UTC is 2025-08-27 23:23:29."}</TOOL_RESULT>

[SOPR]The tool result provides the exact UTC time: 2025-08-27 23:23:29. This is the official start moment, which can now be reported to the user.[EOPR]
[EOR]
<SAY>您馬拉松的官方起跑時刻是 **2025-08-27 23:23:29 UTC**。請直接將這個時間作為您的參考起跑點；如果您的報名系統或當地要求使用台灣時區，記得要再做換算喔。</SAY>"""

_EXAMPLE_INPUT = json.dumps(
    {
        "language": "zh-TW",
        "context": [],
        "user": "Could you provide the exact current date and time so I can record the official start moment for the marathon?",
        "available_tools": [
            {"name": "time_mcp_server_current_time", "description": "Get the current date and time."}
        ],
        "tool_steps": [
            {
                "order": 1,
                "tool_call_line": '<TOOL_CALL>{"function": {"name": "time_mcp_server_current_time", "arguments": {"format": "YYYY-MM-DD HH:mm:ss"}}}</TOOL_CALL>',
                "tool_result_line": '<TOOL_RESULT>{"name": "time_mcp_server_current_time", "response": "Current UTC time is 2025-08-27 23:23:29, and the time in UTC is 2025-08-27 23:23:29."}</TOOL_RESULT>',
            }
        ],
        "reference_answer": "The official start moment for your marathon is 2025-08-27 23:23:29 UTC.",
    },
    ensure_ascii=False,
    indent=2,
)

_EXAMPLE_OUTPUT = json.dumps(
    {
        "user": "可以幫我查一下現在的確切日期與時間，我想拿來記錄馬拉松的官方起跑時刻嗎？",
        "msg": _EXAMPLE_MSG,
    },
    ensure_ascii=False,
)

DIRECT_OUTPUT_CONTRACT = f"""## Output format (READ CAREFULLY)

You write the WHOLE trajectory yourself and return it as a single string, together with the user's
question translated into Traditional Chinese.

Return valid JSON ONLY, exactly this shape (no markdown, no commentary):

{{
  "user": "<the user's question translated into natural Traditional Chinese (zh-TW)>",
  "msg": "<the complete STITCH-S trajectory as a single string>"
}}

Build `msg` by following the "Canonical chunk cycle" and "Markup" rules above, splicing each step's
`tool_call_line` and `tool_result_line` in VERBATIM where the cycle places <TOOL_CALL> / <TOOL_RESULT>.
String serialization:
- Separate chunks with a blank line.
- Write the closing exactly as `[EOPR]\\n[EOR]\\n<SAY>...`: the final [SOPR] block, then [EOR], then the
  final <SAY>, each on its own line and with no blank line inside the closing.
- If `tool_steps` is empty, `msg` is an opening <SAY> followed by that same closing block, with no
  <TOOL_CALL> / <TOOL_RESULT> in between.

## One complete example

INPUT:
{_EXAMPLE_INPUT}

OUTPUT:
{_EXAMPLE_OUTPUT}
"""


def build_system_prompt() -> str:
    """Combine prompt.txt's STITCH-S rules with the direct output contract.

    prompt.txt owns all the rules (language, markup, chunk cycle, constraints);
    DIRECT_OUTPUT_CONTRACT owns only the {user, msg} output shape, the string
    serialization details, and the worked example. Neither side re-defines the
    other's content, so this is a plain concatenation.
    """
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read().rstrip() + "\n\n" + DIRECT_OUTPUT_CONTRACT


# --------------------------------------------------------------------------- #
# Turn splitting / input preparation
# --------------------------------------------------------------------------- #
def split_turns(messages: list[dict]) -> list[dict]:
    """Group a flat message list into turns.

    A turn = one user message followed by every assistant message up to (but not
    including) the next user message. A leading `system` message is captured as
    extra context for the whole conversation.
    """
    turns: list[dict] = []
    system_text = ""
    current: dict | None = None

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_text = (msg.get("content") or "").strip()
            continue
        if role == "user":
            if current is not None:
                turns.append(current)
            current = {"user": msg.get("content") or "", "assistant_messages": []}
        elif role == "assistant":
            if current is None:
                continue
            current["assistant_messages"].append(msg)

    if current is not None:
        turns.append(current)

    if system_text and turns:
        turns[0]["system"] = system_text
    return turns


def collect_tool_steps(assistant_messages: list[dict]) -> list[dict]:
    """Pair every tool_call with its tool_response across the turn's messages.

    Each step carries the exact verbatim markup lines the model must embed. Tool
    calls without a recorded response are dropped (a STITCH-S trajectory needs a
    result and we must never fabricate one).
    """
    steps: list[dict] = []
    for msg in assistant_messages:
        calls = msg.get("tool_calls") or []
        responses = msg.get("tool_responses") or []
        for i, call in enumerate(calls):
            if i >= len(responses) or responses[i] is None:
                continue
            response = responses[i]
            fn = call.get("function", call)
            call_json = json.dumps(call, ensure_ascii=False)
            result_json = json.dumps(response, ensure_ascii=False)
            steps.append(
                {
                    "name": fn.get("name"),
                    "tool_call_line": f"<TOOL_CALL>{call_json}</TOOL_CALL>",
                    "tool_result_line": f"<TOOL_RESULT>{result_json}</TOOL_RESULT>",
                }
            )
    return steps


def reference_answer_of(assistant_messages: list[dict]) -> str:
    """The last non-empty assistant content is the ground-truth final answer."""
    for msg in reversed(assistant_messages):
        content = msg.get("content")
        if content and content.strip():
            return content.strip()
    return ""


def available_tools_of(record: dict) -> list[dict]:
    out = []
    for t in record.get("tools") or []:
        fn = t.get("function", t)
        out.append({"name": fn.get("name"), "description": fn.get("description", "")})
    return out


def build_model_input(
    turn: dict,
    tool_steps: list[dict],
    reference_answer: str,
    available_tools: list[dict],
    context: list[dict],
) -> dict:
    """Assemble the per-turn INPUT object handed to the model."""
    payload = {
        "language": "zh-TW",
        "context": context,
        "user": turn["user"],
        "available_tools": available_tools,
        "tool_steps": [
            {
                "order": i + 1,
                "tool_call_line": s["tool_call_line"],
                "tool_result_line": s["tool_result_line"],
            }
            for i, s in enumerate(tool_steps)
        ],
        "reference_answer": reference_answer,
    }
    if turn.get("system"):
        payload["system"] = turn["system"]
    return payload


# --------------------------------------------------------------------------- #
# Light validation (warn only — we never rewrite the model's trajectory)
# --------------------------------------------------------------------------- #
_SAY_RE = re.compile(r"<SAY>(.*?)</SAY>", re.S)


def last_say(msg: str) -> str:
    matches = _SAY_RE.findall(msg)
    return matches[-1].strip() if matches else ""


def validate_msg(msg: str, tool_steps: list[dict], where: str) -> None:
    if not msg.lstrip().startswith("<SAY>"):
        print(f"  [warn] {where}: msg does not start with <SAY>", file=sys.stderr)
    for s in tool_steps:
        if s["tool_call_line"] not in msg:
            print(
                f"  [warn] {where}: tool_call for {s['name']} not reproduced verbatim",
                file=sys.stderr,
            )
        if s["tool_result_line"] not in msg:
            print(
                f"  [warn] {where}: tool_result for {s['name']} not reproduced verbatim",
                file=sys.stderr,
            )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def process_record(record: dict, system_prompt: str, model: str) -> list[dict]:
    """Turn one multi-turn conversation into a list of single-turn output rows."""
    rows: list[dict] = []
    rec_id = record.get("id", "unknown")
    source = record.get("source", "unknown")
    tools = available_tools_of(record)
    turns = split_turns(record.get("messages", []))

    context: list[dict] = []  # accumulated zh-TW context from earlier turns

    for t_idx, turn in enumerate(turns):
        if not (turn.get("user") or "").strip():
            continue
        tool_steps = collect_tool_steps(turn["assistant_messages"])
        reference_answer = reference_answer_of(turn["assistant_messages"])

        if not tool_steps and not reference_answer:
            # Keep a turn if it has EITHER tool steps OR a reference answer. Tool
            # call+result alone is enough (no assistant text needed — constraint #10
            # synthesizes the closing). Only skip when there is nothing to ground a
            # trajectory on (e.g. apigen: tool_calls with no recorded result and no
            # text answer). Skip rather than hallucinate.
            print(
                f"  [warn] turn {t_idx} of {rec_id}: no tool results and no "
                f"reference answer, skipping",
                file=sys.stderr,
            )
            continue

        model_input = build_model_input(
            turn, tool_steps, reference_answer, tools, list(context)
        )
        prompt = (
            "INPUT:\n"
            + json.dumps(model_input, ensure_ascii=False, indent=2)
            + "\n\nOUTPUT:\n"
        )

        where = f"turn {t_idx} of {rec_id}"
        try:
            raw = call_gemma(prompt, system=system_prompt, model=model, fmt="json")
            obj = _extract_json(raw)
        except Exception as exc:  # noqa: BLE001 - keep the run going
            print(f"  [warn] {where} failed: {exc}", file=sys.stderr)
            continue

        user_zh = (obj.get("user") or turn["user"]).strip()
        msg = (obj.get("msg") or "").strip()
        if not msg:
            print(f"  [warn] {where}: empty msg, skipping", file=sys.stderr)
            continue

        validate_msg(msg, tool_steps, where)

        rows.append(
            {
                "id": f"{rec_id}::turn{t_idx}",
                "source": source,
                "user": user_zh,
                "msg": msg,
            }
        )

        # Feed this turn forward as zh-TW context for the next turn.
        context.append({"user": user_zh, "assistant": last_say(msg)})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=DATA_PATH, help="Input records (.jsonl or .parquet)."
    )
    parser.add_argument("--out", default=OUT_PATH, help="Single-turn .jsonl output.")
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMMA_MODEL", "gemma-4-26b-a4b-it"),
        help="Gemini API model id to use.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N records."
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Skip the first START records."
    )
    args = parser.parse_args()

    system_prompt = build_system_prompt()

    records = load_records(args.data)

    selected = records[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    total_rows = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for r_idx, record in enumerate(selected, start=args.start):
            rec_id = record.get("id", f"record{r_idx}")
            print(f"[{r_idx}] {rec_id} ...", file=sys.stderr)
            try:
                rows = process_record(record, system_prompt, args.model)
            except Exception:  # noqa: BLE001
                print(
                    f"  [error] record {rec_id} crashed:\n" + traceback.format_exc(),
                    file=sys.stderr,
                )
                continue
            for row in rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
            total_rows += len(rows)
            print(f"    -> {len(rows)} single-turn rows", file=sys.stderr)

    print(f"Done. Wrote {total_rows} single-turn rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
