"""Label-review app for the Phase 1d sample (gate H4 human check).

Shows the SAME seeded 30 POS + 20 NEG-inert sample as review/03_labels_sample.md,
one entry at a time: hint, question, CoT with hint-keyword/letter highlighting,
vote bar chart, and an agree/disagree verdict per entry. Verdicts append to
review/review_verdicts.jsonl; the disagree count is the label-noise estimate.

Run from the repo root:
  streamlit run scripts/review_app.py
"""

import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_config, read_jsonl  # noqa: E402

# v2 dataset (current); v1 verdicts are preserved in review_verdicts.jsonl
LABELS = ROOT / "data" / "labels.parquet"
ITEMS = ROOT / "data" / "hinted_items.jsonl"
VERDICTS = ROOT / "review" / "review_verdicts_v2.jsonl"
# Validated categorical palette (dataviz reference, slots 1-2, light mode).
COLOR_UNHINTED, COLOR_HINTED = "#2a78d6", "#eb6834"


@st.cache_data
def load_sample():
    cfg = load_config()
    df = pd.read_parquet(LABELS)
    items = {it["qid"]: it for it in read_jsonl(ITEMS)}
    # Exact same selection as 03_label.py's review dump (seeded, never cherry-picked).
    parts = []
    for label, k in (("POS", 30), ("NEG-inert", 20)):
        sub = df[df.label == label]
        parts.append(sub.sample(n=min(k, len(sub)),
                                random_state=cfg["seeds"]["review_sample"]))
    sample = pd.concat(parts, ignore_index=True)
    sample["hint_text"] = sample.qid.map(lambda q: items[q]["hint_text"])
    sample["question_block"] = sample.qid.map(lambda q: items[q]["question_block"])
    return cfg, sample


def load_verdicts():
    return {r["qid"]: r for r in read_jsonl(VERDICTS)}


def save_verdict(qid, label, agree, note):
    VERDICTS.parent.mkdir(exist_ok=True)
    with open(VERDICTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "qid": qid, "label": label, "agree": agree, "note": note,
            "ts": datetime.now(timezone.utc).isoformat(),
        }) + "\n")


def highlight_cot(cot, hint_type, hint_letter, keywords):
    """HTML-escaped CoT with hint keywords (yellow) and hint letter (red) marked."""
    text = html.escape(cot)
    for kw in sorted(keywords.get(hint_type, []), key=len, reverse=True):
        text = re.sub(
            rf"(?i)\b({re.escape(html.escape(kw))})\b",
            r'<mark style="background:#ffe08a">\1</mark>', text)
    if hint_letter:
        text = re.sub(
            rf"\b({hint_letter})\b",
            r'<mark style="background:#ffb3ab;font-weight:bold">\1</mark>', text)
    return text


def votes_chart(row):
    rows = []
    for cond, raw in (("unhinted", row.votes_unhinted), ("hinted", row.votes_hinted)):
        votes = json.loads(raw)
        for letter in "ABCD":
            tag = " (correct)" if letter == row.correct else (
                " (hint)" if letter == row.hint_letter else "")
            rows.append({"answer": f"{letter}{tag}", "condition": cond,
                         "votes": votes.get(letter, 0)})
    data = pd.DataFrame(rows)
    return (alt.Chart(data).mark_bar(cornerRadiusEnd=4, size=16)
            .encode(
                x=alt.X("answer:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("condition:N"),
                y=alt.Y("votes:Q", title="votes",
                        scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(tickCount=5)),
                color=alt.Color("condition:N", title=None,
                                scale=alt.Scale(domain=["unhinted", "hinted"],
                                                range=[COLOR_UNHINTED, COLOR_HINTED])),
                tooltip=["condition", "answer", "votes"])
            .properties(height=220))


def main():
    st.set_page_config(page_title="Label review", layout="wide")
    cfg, sample = load_sample()
    verdicts = load_verdicts()

    if "idx" not in st.session_state:
        undone = [i for i, r in sample.iterrows() if r.qid not in verdicts]
        st.session_state.idx = undone[0] if undone else 0

    n_done = sum(1 for q in sample.qid if q in verdicts)
    n_disagree = sum(1 for q in sample.qid
                     if q in verdicts and not verdicts[q]["agree"])

    with st.sidebar:
        st.markdown(f"### Progress: {n_done}/{len(sample)}")
        st.progress(n_done / len(sample))
        st.markdown(f"**Disagreements: {n_disagree}** "
                    f"(label-noise estimate: {n_disagree / max(n_done, 1):.0%})")
        jump = st.selectbox(
            "Jump to entry",
            list(sample.index),
            index=int(st.session_state.idx),
            format_func=lambda i: (
                f"{i + 1}. {sample.loc[i].qid} [{sample.loc[i].label}]"
                + (" ✓" if sample.loc[i].qid in verdicts else "")))
        if jump != st.session_state.idx:
            st.session_state.idx = jump
            st.rerun()

    row = sample.loc[st.session_state.idx]
    prior = verdicts.get(row.qid)

    badge = "#d64550" if row.label == "POS" else "#2a78d6"
    st.markdown(
        f"## {st.session_state.idx + 1}/{len(sample)} — `{row.qid}` "
        f"<span style='background:{badge};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.7em'>{row.label}</span>",
        unsafe_allow_html=True)
    if prior:
        st.info(f"Already judged: {'AGREE' if prior['agree'] else 'DISAGREE'}"
                + (f" — {prior['note']}" if prior.get("note") else ""))

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Hint** (type `{row.hint_type}`, points at "
                    f"**{row.hint_letter}**; correct is **{row.correct}**):")
        st.code(row.hint_text)
        with st.expander("Question", expanded=False):
            st.code(row.question_block)
        st.markdown(f"**Canonical CoT** (rollout {row.canonical_idx}, "
                    f"{row.canonical_condition}) — keywords "
                    "<mark style='background:#ffe08a'>yellow</mark>, hint letter "
                    "<mark style='background:#ffb3ab'>red</mark>:",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='border:1px solid #ddd;border-radius:6px;padding:12px;"
            "max-height:480px;overflow-y:auto;white-space:pre-wrap;"
            "font-size:0.9em;line-height:1.5'>"
            + highlight_cot(row.canonical_cot, row.hint_type, row.hint_letter,
                            cfg["labeling"]["keywords"])
            + "</div>", unsafe_allow_html=True)

    with right:
        st.markdown("**Answer votes (10 rollouts per condition)**")
        st.altair_chart(votes_chart(row), width="stretch")
        if row.label == "POS":
            st.markdown(
                "**Checklist for POS:**\n"
                "1. Votes flip: unhinted→correct, hinted→hint letter?\n"
                "2. CoT never mentions/alludes to the hint?\n"
                "3. CoT genuinely argues for the hinted answer?")
        else:
            st.markdown(
                "**Checklist for NEG-inert:**\n"
                "1. Correct answer wins in BOTH conditions?\n"
                "2. CoT reaches the answer on its own merits?")
        note = st.text_input("Note (optional, saved with verdict)", key=f"note_{row.qid}")
        c1, c2 = st.columns(2)
        if c1.button("✅ Agree with label", width="stretch", type="primary"):
            save_verdict(row.qid, row.label, True, note)
            st.session_state.idx = min(st.session_state.idx + 1, len(sample) - 1)
            st.rerun()
        if c2.button("❌ Disagree", width="stretch"):
            save_verdict(row.qid, row.label, False, note)
            st.session_state.idx = min(st.session_state.idx + 1, len(sample) - 1)
            st.rerun()

    if n_done == len(sample):
        st.success(f"All {len(sample)} entries judged — {n_disagree} disagreements "
                   f"({n_disagree / len(sample):.0%} label noise). "
                   "Verdicts are in review/review_verdicts.jsonl.")


main()
