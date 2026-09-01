"""Phase 1a — build the base question pool.

Two stages:
  python src/00_filter_questions.py pull      # download + format + seeded subsample (no GPU)
  python src/00_filter_questions.py prepass   # unhinted accuracy pre-pass (needs vLLM server)

pull    -> data/base_questions_raw.jsonl  (candidate pool)
prepass -> data/rollouts/prepass.jsonl    (per-item checkpoint, resume-safe)
        -> data/base_questions.jsonl      (items answered correctly >= 2/3 unhinted)
"""

import asyncio
import sys

from common import (LETTERS, ROOT, append_jsonl, chat_batch, item_rng, load_config,
                    majority, parse_answer, policy_extra_body, question_block,
                    read_jsonl, rng, vllm_client, write_jsonl)

RAW = ROOT / "data" / "base_questions_raw.jsonl"
PREPASS = ROOT / "data" / "rollouts" / "prepass.jsonl"
KEPT = ROOT / "data" / "base_questions.jsonl"


def pull(cfg):
    from datasets import load_dataset

    p1 = cfg["phase1"]
    items = []

    for subject in p1["mmlu_subjects"]:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for i, row in enumerate(ds):
            if len(row["choices"]) != 4:
                continue
            items.append({
                "qid": f"mmlu_{subject}_{i}",
                "source": "mmlu",
                "subject": subject,
                "question": row["question"],
                "choices": row["choices"],
                "correct": LETTERS[row["answer"]],
            })
        print(f"mmlu/{subject}: {len(ds)} items", flush=True)

    if p1["use_gpqa"]:
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            for i, row in enumerate(ds):
                qid = f"gpqa_diamond_{i}"
                options = [row["Correct Answer"]] + [row[f"Incorrect Answer {k}"] for k in (1, 2, 3)]
                options = [o.strip() for o in options]
                r = item_rng(cfg["seeds"]["gpqa_shuffle"], qid)
                order = list(range(4))
                r.shuffle(order)
                items.append({
                    "qid": qid,
                    "source": "gpqa",
                    "subject": "gpqa_diamond",
                    "question": row["Question"],
                    "choices": [options[j] for j in order],
                    "correct": LETTERS[order.index(0)],
                })
            print(f"gpqa_diamond: {len(ds)} items", flush=True)
        except Exception as e:  # noqa: BLE001 — gated dataset; MMLU alone is fine
            print(f"WARNING: GPQA unavailable ({e}). Continuing with MMLU only.", flush=True)

    r = rng(cfg["seeds"]["question_pool"])
    r.shuffle(items)
    pool = items[: p1["candidate_pool_size"]]
    write_jsonl(RAW, pool)
    print(f"wrote {len(pool)} / {len(items)} candidates -> {RAW}", flush=True)


def prepass(cfg):
    p1 = cfg["phase1"]
    pool = read_jsonl(RAW)
    if not pool:
        raise SystemExit(f"{RAW} missing — run the pull stage first")
    by_qid = {it["qid"]: it for it in pool}

    done = {rec["qid"] for rec in read_jsonl(PREPASS)}
    order = sorted(by_qid)
    rng(cfg["seeds"]["prepass_order"]).shuffle(order)

    # Early stop: no need to keep burning rollouts once the kept target is hit.
    kept_so_far = sum(
        rec["n_correct"] >= p1["prepass_keep_min_correct"] for rec in read_jsonl(PREPASS)
    )
    todo = [q for q in order if q not in done]
    print(f"{len(done)} done, {kept_so_far} kept so far, {len(todo)} remaining", flush=True)

    client = vllm_client(cfg)
    system = cfg["prompting"]["system"]

    def on_result(qid, texts):
        item = by_qid[qid]
        answers = [parse_answer(t) for t in texts] if texts else []
        rec = {
            "qid": qid,
            "answers": answers,
            "n_correct": sum(a == item["correct"] for a in answers),
            "parse_failures": sum(a is None for a in answers),
            "rollouts": texts or [],
        }
        append_jsonl(PREPASS, rec)

    batch_size = 200
    for start in range(0, len(todo), batch_size):
        kept = sum(
            rec["n_correct"] >= p1["prepass_keep_min_correct"] for rec in read_jsonl(PREPASS)
        )
        if kept >= p1["target_kept_items"]:
            print(f"target of {p1['target_kept_items']} kept items reached — stopping early", flush=True)
            break
        chunk = todo[start: start + batch_size]
        jobs = [
            (qid, [{"role": "system", "content": system},
                   {"role": "user", "content": question_block(by_qid[qid])}])
            for qid in chunk
        ]
        asyncio.run(chat_batch(
            client, cfg["models"]["policy"], jobs,
            temperature=p1["temperature"], max_tokens=p1["max_tokens"],
            n=p1["prepass_rollouts"], concurrency=cfg["serving"]["max_concurrency"],
            on_result=on_result, extra_body=policy_extra_body(cfg),
        ))
        print(f"prepass progress: {start + len(chunk)}/{len(todo)}", flush=True)

    records = read_jsonl(PREPASS)
    kept_items = [
        # prepass margin travels with the item (used to analyze hint-following
        # vs unhinted confidence).
        {**by_qid[rec["qid"]], "prepass_correct": rec["n_correct"],
         "prepass_n": len(rec["answers"])}
        for rec in records
        if rec["qid"] in by_qid and rec["n_correct"] >= p1["prepass_keep_min_correct"]
    ]
    total_parse_failures = sum(rec["parse_failures"] for rec in records)
    write_jsonl(KEPT, kept_items)
    print(f"kept {len(kept_items)} / {len(records)} items "
          f"({total_parse_failures} parse failures) -> {KEPT}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = load_config()
    if stage == "pull":
        pull(cfg)
    elif stage == "prepass":
        prepass(cfg)
    else:
        raise SystemExit("usage: python src/00_filter_questions.py {pull|prepass}")
