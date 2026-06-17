"""Call Gemma 4 through the Google Gemini API (Generative Language API).

The API key is read from the GOOGLE_API_KEY entry in the project `.env` file
(or from the environment). The default model is `gemma-4-31b-it` — a strong,
currently-healthy teacher for generating SFT data. (`gemma-4-26b-a4b-it`, the
e4b-equivalent, is also available via --model / GEMMA_MODEL but has been throwing
server-side 500s; the retry logic below tolerates transient blips on either.)

Note: Gemma 4 is a *thinking* model. Each response contains a hidden reasoning
part (`thought: true`) followed by the real answer part; `call_gemma` returns
only the answer text. Thinking tokens count against `max_output_tokens`, so the
budget is auto-bumped if a response is truncated (finishReason=MAX_TOKENS).

Usage as a library:
    from callGemma import call_gemma, call_gemma_json
    text = call_gemma("hello", system="You are helpful.")
    obj  = call_gemma_json(prompt, system=prompt_txt)

Quick smoke test:
    python callGemma.py "用繁體中文跟我打聲招呼。"
"""

from __future__ import annotations

import os
import sys
import json
import time
import random

import requests

GEMINI_BASE = os.environ.get(
    "GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta"
)
DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-31b-it")
MODEL_OUTPUT_LIMIT = 32768  # both gemma-4 models cap output at 32k tokens

# Status codes worth retrying (transient server / rate issues).
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def _load_api_key() -> str:
    """Read GOOGLE_API_KEY from the environment, falling back to ./.env."""
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == "GOOGLE_API_KEY":
                    return value.strip().strip('"').strip("'")
    raise RuntimeError(
        "GOOGLE_API_KEY not found. Add it to the .env file or the environment."
    )


def _backoff(attempt: int, retry_after: float | None = None) -> None:
    """Exponential backoff with jitter, capped at ~60s.

    If the server sent a Retry-After hint, honour it (it knows its own load).
    """
    delay = min(2.0 * (2 ** attempt), 60.0) + random.uniform(0, 1.0)
    if retry_after is not None:
        delay = max(delay, retry_after)
    time.sleep(delay)


def _retry_after_seconds(resp: requests.Response) -> float | None:
    """Parse a Retry-After header (seconds form), if present."""
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def call_gemma(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    *,
    temperature: float = 0.7,
    top_k: int = 64,
    top_p: float = 0.95,
    max_output_tokens: int = 8192,
    fmt: str | None = None,
    timeout: int = 600,
    max_retries: int = 8,
) -> str:
    """Send a request to Gemma 4 via the Gemini API and return the answer text.

    Retries transient API errors (5xx/429) with exponential backoff, and bumps
    the output-token budget if a response is truncated by the thinking tokens.

    Args:
        prompt:   The user message.
        system:   Optional system instruction (e.g. the contents of prompt.txt).
        model:    Gemini API model id (e.g. "gemma-4-31b-it").
        fmt:      Pass "json" to constrain the answer to valid JSON.
        max_output_tokens: starting budget; doubled (up to 32k) on truncation.

    Returns:
        The concatenated non-thought answer text.
    """
    key = _load_api_key()
    url = f"{GEMINI_BASE}/models/{model}:generateContent?key={key}"
    budget = min(max_output_tokens, MODEL_OUTPUT_LIMIT)
    last_err: str | None = None

    attempt = 0
    while attempt < max_retries:
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topK": top_k,
                "topP": top_p,
                "maxOutputTokens": budget,
            },
        }
        if system:
            body["system_instruction"] = {"parts": [{"text": system}]}
        if fmt == "json":
            body["generationConfig"]["responseMimeType"] = "application/json"

        try:
            resp = requests.post(url, json=body, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            last_err = f"request error: {exc}"
            _backoff(attempt)
            attempt += 1
            continue

        if resp.status_code in _TRANSIENT_STATUS:
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            _backoff(attempt, _retry_after_seconds(resp))
            attempt += 1
            continue
        if resp.status_code != 200:
            # Non-transient (e.g. 400/403) — don't waste retries.
            raise RuntimeError(
                f"Gemini API error {resp.status_code}: {resp.text[:500]}"
            )

        answer, finish = _answer_and_finish(resp.json())
        if answer:
            return answer

        # Empty answer: if truncated by thinking, grow the budget and retry.
        if finish == "MAX_TOKENS" and budget < MODEL_OUTPUT_LIMIT:
            budget = min(budget * 2, MODEL_OUTPUT_LIMIT)
            last_err = "empty answer (MAX_TOKENS); raised token budget"
            attempt += 1
            continue

        last_err = f"empty answer (finishReason={finish})"
        _backoff(attempt)
        attempt += 1

    raise RuntimeError(
        f"call_gemma failed after {max_retries} attempts for model {model}. "
        f"Last issue: {last_err}"
    )


def _answer_and_finish(data: dict) -> tuple[str, str | None]:
    """Return (answer_text, finishReason); answer excludes the thought part."""
    candidates = data.get("candidates") or []
    if not candidates:
        return "", None
    cand = candidates[0]
    finish = cand.get("finishReason")
    parts = (cand.get("content") or {}).get("parts") or []
    answer = "".join(
        p.get("text", "") for p in parts if not p.get("thought")
    ).strip()
    return answer, finish


def call_gemma_json(prompt: str, system: str | None = None, **kwargs) -> dict:
    """Like `call_gemma` but parse the reply as JSON (asks for JSON mode)."""
    kwargs.setdefault("fmt", "json")
    raw = call_gemma(prompt, system=system, **kwargs)
    return _extract_json(raw)


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a single JSON object from a model reply."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]) or "用繁體中文跟我打聲招呼。"
    print(call_gemma(user_prompt))
