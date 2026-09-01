"""Phase 1c — batched rollouts, both conditions, via the vLLM server.

Per item and condition (hinted / unhinted): 10 rollouts, temperature 0.7, full
CoT. One jsonl line per (item, condition), appended the moment the request
finishes — a crash loses at most in-flight requests. Rerunning skips completed
(qid, condition) pairs.

Run on the pod as an overnight background job:
  nohup python src/02_generate.py > logs/02_generate.log 2>&1 &

data/hinted_items.jsonl -> data/rollouts/rollouts.jsonl
"""

import asyncio

from common import (ROOT, append_jsonl, chat_batch, load_config, parse_answer,
                    policy_extra_body, read_jsonl, vllm_client)

HINTED = ROOT / "data" / "hinted_items.jsonl"
OUT = ROOT / "data" / "rollouts" / "rollouts.jsonl"
CONDITIONS = ["unhinted", "hinted"]


def build_user_prompt(item, condition):
    if condition == "hinted":
        return item["hint_text"] + "\n\n" + item["question_block"]
    return item["question_block"]


def main():
    cfg = load_config()
    p1 = cfg["phase1"]
    items = read_jsonl(HINTED)
    if not items:
        raise SystemExit(f"{HINTED} missing — run 01_build_hinted.py first")
    by_qid = {it["qid"]: it for it in items}

    done = {(rec["qid"], rec["condition"]) for rec in read_jsonl(OUT)}
    todo = [
        (qid, cond)
        for qid in sorted(by_qid)
        for cond in CONDITIONS
        if (qid, cond) not in done
    ]
    print(f"{len(done)} item-conditions done, {len(todo)} remaining", flush=True)

    client = vllm_client(cfg)
    system = cfg["prompting"]["system"]

    def on_result(key, texts):
        qid, condition = key
        if texts is None:
            return  # logged by chat_batch; picked up on the next resume
        answers = [parse_answer(t) for t in texts]
        append_jsonl(OUT, {
            "qid": qid,
            "condition": condition,
            "rollouts": texts,
            "answers": answers,
            "parse_failures": sum(a is None for a in answers),
        })

    batch_size = 100
    for start in range(0, len(todo), batch_size):
        chunk = todo[start: start + batch_size]
        jobs = [
            ((qid, cond), [{"role": "system", "content": system},
                           {"role": "user", "content": build_user_prompt(by_qid[qid], cond)}])
            for qid, cond in chunk
        ]
        asyncio.run(chat_batch(
            client, cfg["models"]["policy"], jobs,
            temperature=p1["temperature"], max_tokens=p1["max_tokens"],
            n=p1["rollouts_per_condition"], concurrency=cfg["serving"]["max_concurrency"],
            on_result=on_result, extra_body=policy_extra_body(cfg),
        ))
        print(f"progress: {start + len(chunk)}/{len(todo)} item-conditions", flush=True)

    records = read_jsonl(OUT)
    failures = sum(rec["parse_failures"] for rec in records)
    total = sum(len(rec["answers"]) for rec in records)
    print(f"done: {len(records)} item-conditions, {total} rollouts, "
          f"{failures} parse failures ({failures / max(total, 1):.1%})", flush=True)


if __name__ == "__main__":
    main()
