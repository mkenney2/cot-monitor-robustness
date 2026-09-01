"""Shared utilities for the monitor-reliability pipeline.

Conventions:
- All randomness goes through rng(seed) or item_rng(seed, qid); seeds live in config.yaml.
- All generation goes through chat_batch(), which checkpoints incrementally via a
  callback so a crashed run never loses completed items.
"""

import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LETTERS = ["A", "B", "C", "D"]

# The model is instructed to end with "Answer: X"; we take the LAST match so a
# CoT that discusses candidate answers mid-stream doesn't confuse the parser.
ANSWER_RE = re.compile(r"answer\s*:\s*\(?\s*([ABCD])\s*\)?", re.IGNORECASE)


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rng(seed):
    return random.Random(seed)


def item_rng(seed, qid):
    # Random() seeded with a string is stable across processes (unlike hash()).
    return random.Random(f"{seed}:{qid}")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_answer(text):
    """Strict parser: last 'Answer: X' occurrence, or None (a counted parse failure)."""
    matches = ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else None


def question_block(item):
    lines = [item["question"].strip(), ""]
    for letter, choice in zip(LETTERS, item["choices"]):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def majority(letters):
    """Majority letter among parsed answers and full vote counts.

    Returns (majority_letter_or_None, counts_dict). Majority requires a strict
    plurality: ties return None.
    """
    parsed = [x for x in letters if x is not None]
    counts = dict(Counter(parsed))
    if not parsed:
        return None, counts
    ranked = Counter(parsed).most_common(2)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, counts
    return ranked[0][0], counts


def load_hint_template(hint_type):
    return (ROOT / "data" / "hints" / f"{hint_type}.txt").read_text(encoding="utf-8").strip()


def vllm_client(cfg):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=cfg["serving"]["vllm_base_url"],
        api_key=cfg["serving"]["vllm_api_key"],
        timeout=cfg["serving"]["request_timeout_s"],
    )


def openrouter_client(cfg):
    import os

    from openai import AsyncOpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    return AsyncOpenAI(
        base_url=cfg["serving"]["openrouter_base_url"],
        api_key=key,
        timeout=cfg["serving"]["request_timeout_s"],
    )


def policy_extra_body(cfg):
    """vLLM-only request options for the policy model (never for OpenRouter)."""
    return {"chat_template_kwargs":
            {"enable_thinking": cfg["prompting"]["enable_thinking"]}}


async def chat_batch(client, model, jobs, *, temperature, max_tokens, n,
                     concurrency, on_result, extra_body=None):
    """Run chat jobs concurrently with retries and incremental checkpointing.

    jobs: list of (key, messages). For each job, on_result(key, texts) is called
    as soon as its request finishes (texts is a list of n completion strings, or
    None if all retries failed — the caller logs it and moves on).
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(key, messages):
        async with sem:
            for attempt in range(4):
                try:
                    resp = await client.chat.completions.create(
                        model=model, messages=messages, temperature=temperature,
                        max_tokens=max_tokens, n=n, extra_body=extra_body,
                    )
                    on_result(key, [c.message.content or "" for c in resp.choices])
                    return
                except Exception as e:  # noqa: BLE001 — retry then surface
                    if attempt == 3:
                        print(f"[FAIL] {key}: {e}", flush=True)
                        on_result(key, None)
                        return
                    await asyncio.sleep(2 ** attempt * 5)

    await asyncio.gather(*(one(k, m) for k, m in jobs))
