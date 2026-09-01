"""Phase 1b — assign each kept item a hint type + hint letter and render the hint.

The hint always points at a specific WRONG answer (seeded pick among the three
wrong options) so hint-following is behaviorally detectable. Hint types are
assigned over a seeded shuffle with counts from config hint_type_weights
(largest-remainder rounding, deterministic).

data/base_questions.jsonl -> data/hinted_items.jsonl
The hint text is stored as its own field (not baked into a prompt string) so the
hint region can be redacted for the hint-blind judge variant in Phase 2.
"""

from common import (LETTERS, ROOT, item_rng, load_config, load_hint_template,
                    question_block, read_jsonl, rng, write_jsonl)

KEPT = ROOT / "data" / "base_questions.jsonl"
OUT = ROOT / "data" / "hinted_items.jsonl"


def main():
    cfg = load_config()
    items = read_jsonl(KEPT)
    if not items:
        raise SystemExit(f"{KEPT} missing — run 00_filter_questions.py first")

    weights = cfg["phase1"]["hint_type_weights"]
    hint_types = sorted(weights)
    templates = {t: load_hint_template(t) for t in hint_types}

    order = sorted(range(len(items)), key=lambda i: items[i]["qid"])
    rng(cfg["seeds"]["hint_assignment"]).shuffle(order)

    # Counts per type: largest-remainder rounding of the configured weights.
    n = len(items)
    quotas = {t: weights[t] * n for t in hint_types}
    counts = {t: int(quotas[t]) for t in hint_types}
    for t in sorted(hint_types, key=lambda t: quotas[t] - counts[t],
                    reverse=True)[: n - sum(counts.values())]:
        counts[t] += 1
    assignment = [t for t in hint_types for _ in range(counts[t])]

    out = []
    for pos, idx in enumerate(order):
        item = items[idx]
        hint_type = assignment[pos]
        wrong = [x for x in LETTERS if x != item["correct"]]
        hint_letter = item_rng(cfg["seeds"]["hint_assignment"], item["qid"]).choice(wrong)
        out.append({
            **item,
            "hint_type": hint_type,
            "hint_letter": hint_letter,
            "hint_text": templates[hint_type].format(letter=hint_letter),
            "question_block": question_block(item),
        })

    out.sort(key=lambda x: x["qid"])
    write_jsonl(OUT, out)

    per_type = {t: sum(x["hint_type"] == t for x in out) for t in hint_types}
    print(f"wrote {len(out)} hinted items -> {OUT}")
    print(f"hint-type balance: {per_type}")


if __name__ == "__main__":
    main()
