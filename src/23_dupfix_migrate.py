"""One-off migration for the duplicate-qid collision (see STATUS.md, 2026-09-01/02).

16 qids in the analysis set carry two canonical transcripts (hinted -> POS or
NEG-inert; unhinted -> NEG-clean). Caches were keyed by qid only, so each pair
shared one activation file / one judge record / one lens-LLM record. The fix
keys everything by the transcript key `tkey` (phase2_common.transcript_key):
qid for hinted transcripts, `{qid}__clean` for unhinted ones.

This script migrates what exists so nothing already computed is recomputed
unnecessarily, then VERIFIES every activation file against the transcript it
now claims to hold (token-stream match), and reports exactly what is missing
(to be filled by the resume-safe cache/judge stages):

  activations/ (16/32/48) and activations_late/ (56/60/62):
    every NEG-clean row's file {qid}.npz is renamed {qid}__clean.npz IF its
    token stream matches the unhinted transcript; a file whose stream matches
    the hinted transcript keeps the {qid}.npz name. Mismatches abort.
  judge_raw.jsonl, jlens_llm_raw.jsonl, jlens_llm_late_raw.jsonl:
    records gain a `tkey`; for NON-duplicated qids the row is unambiguous
    (qid__clean for the NEG-clean row, qid otherwise); records for the 16
    duplicated qids are DROPPED (provenance unknowable) so both rows get
    re-scored. A backup copy of each jsonl is written next to it.

Usage (pod): python src/23_dupfix_migrate.py [--dry-run]
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from common import read_jsonl  # noqa: E402
from phase2_common import load_dataset  # noqa: E402

ACT_DIRS = [ROOT / "data" / "activations", ROOT / "data" / "activations_late"]
JSONL_CACHES = [ROOT / "data" / "rollouts" / "judge_raw.jsonl",
                ROOT / "data" / "rollouts" / "jlens_llm_raw.jsonl",
                ROOT / "data" / "rollouts" / "jlens_llm_late_raw.jsonl"]

_squash = lambda t: re.sub(r"[^A-Za-z0-9]", "", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from common import load_config
    tok = AutoTokenizer.from_pretrained(load_config()["models"]["policy"])

    df = load_dataset()
    dups = set(df[df.qid.duplicated(keep=False)].qid)
    by_qid = {q: g for q, g in df.groupby("qid")}
    print(f"{len(df)} rows, {df.qid.nunique()} unique qids, {len(dups)} duplicated", flush=True)

    def which_transcript(path, qid):
        """Label of the transcript whose text matches the cached token stream."""
        d = np.load(path)
        head = _squash(tok.decode(d["token_ids"][:300]))[:250]
        hits = [r.label for _, r in by_qid[qid].iterrows() if head and head in _squash(r.canonical_cot)]
        return hits

    report = {}
    for act_dir in ACT_DIRS:
        if not act_dir.exists():
            continue
        renamed, problems = [], []
        # Pass 1: every legacy {qid}.npz whose token stream is the UNHINTED transcript
        # moves to {qid}__clean.npz (for non-duplicated NEG-clean rows and for the
        # duplicated qids whose cache happened to hold the unhinted transcript).
        for q in sorted(df.qid.unique()):
            legacy = act_dir / f"{q}.npz"
            if not legacy.exists():
                continue
            hits = which_transcript(legacy, q)
            if hits == ["NEG-clean"]:
                target = act_dir / f"{q}__clean.npz"
                if target.exists():
                    problems.append((legacy.name, "unhinted stream but __clean file already exists"))
                    continue
                if not args.dry_run:
                    legacy.rename(target)
                renamed.append(legacy.name)
            elif len(hits) != 1:
                problems.append((legacy.name, f"ambiguous/empty text match: {hits}"))
        # Pass 2: verify every file that exists under a tkey holds that transcript.
        present = 0
        for _, row in df.iterrows():
            f = act_dir / f"{row.tkey}.npz"
            if args.dry_run and f"{row.qid}.npz" in renamed:   # not yet renamed in a dry run
                if row.label == "NEG-clean":
                    f = act_dir / f"{row.qid}.npz"
                else:
                    continue                                   # would be gone after the rename
            if not f.exists():
                continue
            present += 1
            hits = which_transcript(f, row.qid)
            if hits != [row.label]:
                problems.append((f.name, f"expected {row.label}, text matches {hits}"))
        def will_exist(r):
            if args.dry_run and f"{r.qid}.npz" in renamed:
                return r.label == "NEG-clean"
            return (act_dir / f"{r.tkey}.npz").exists()
        missing = [r.tkey for _, r in df.iterrows() if not will_exist(r)]
        report[act_dir.name] = {"present_verified": present, "renamed": renamed,
                                "missing_after": missing, "problems": problems}
        print(f"{act_dir.name}: {present} files verified against their transcript, "
              f"{len(renamed)} renamed to __clean, {len(missing)} missing (to be cached): {missing}",
              flush=True)
        if problems:
            print(f"  PROBLEMS: {problems}", flush=True)

    for cache in JSONL_CACHES:
        if not cache.exists():
            continue
        recs = read_jsonl(cache)
        out, dropped, tagged = [], 0, 0
        for r in recs:
            if "tkey" in r:
                out.append(r); continue
            q = r["qid"]
            if q in dups:
                dropped += 1; continue
            rows = by_qid.get(q)
            if rows is None:
                out.append(r); continue          # not in the analysis set; keep as-is
            r["tkey"] = rows.tkey.iloc[0]
            tagged += 1
            out.append(r)
        print(f"{cache.name}: {len(recs)} records -> {len(out)} ({tagged} tagged with tkey, "
              f"{dropped} duplicate-qid records dropped)", flush=True)
        if not args.dry_run:
            shutil.copy2(cache, cache.with_suffix(".pre_dupfix.jsonl"))
            with open(cache, "w", encoding="utf-8") as f:
                for r in out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[cache.name] = {"n_before": len(recs), "n_after": len(out), "dropped": dropped}

    (ROOT / "results" / "dupfix_migration.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    problems = [p for v in report.values() if isinstance(v, dict) for p in v.get("problems", [])]
    if problems:
        raise SystemExit(f"ABORT: {len(problems)} activation files do not match their transcript")
    print("migration report -> results/dupfix_migration.json", flush=True)


if __name__ == "__main__":
    main()
