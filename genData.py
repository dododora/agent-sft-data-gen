"""Generate STITCH-S single-turn training data from the gemma4-agent-sft canon.

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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from callGemma import call_gemma, _extract_json  # noqa: E402

DATA_PATH = os.path.join(HERE, "gemma4-agent-sft", "canonical", "small_data.jsonl")
PROMPT_PATH = os.path.join(HERE, "prompt.txt")
OUT_DIR = os.path.join(HERE, "out")
OUT_PATH = os.path.join(OUT_DIR, "single_turn.jsonl")


def _collapse_tool_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    last_assistant: dict | None = None
    for msg in messages:
        if msg.get("role") == "tool":
            if last_assistant is not None:
                last_assistant.setdefault("tool_responses", []).extend(
                    msg.get("tool_responses") or []
                )
            continue
        msg = dict(msg)
        out.append(msg)
        if msg.get("role") == "assistant":
            last_assistant = msg
    return out


def _parse_record(rec: dict) -> dict:
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
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["id", "source", "messages", "tools"])
        return [_parse_record(row) for row in table.to_pylist()]

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_parse_record(json.loads(line)))
    return records


def build_system_prompt() -> str:
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read().rstrip()


def split_turns(messages: list[dict]) -> list[dict]:
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


def process_record(record: dict, system_prompt: str, model: str) -> list[dict]:
    rows: list[dict] = []
    rec_id = record.get("id", "unknown")
    source = record.get("source", "unknown")
    tools = available_tools_of(record)
    turns = split_turns(record.get("messages", []))

    context: list[dict] = []

    for t_idx, turn in enumerate(turns):
        if not (turn.get("user") or "").strip():
            continue
        tool_steps = collect_tool_steps(turn["assistant_messages"])
        reference_answer = reference_answer_of(turn["assistant_messages"])

        if not tool_steps and not reference_answer:
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
        except Exception as exc:  # noqa: BLE001
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
                "input": model_input,
            }
        )

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
        default=os.environ.get("GEMMA_MODEL", "gemma-4-31b-it"),
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
