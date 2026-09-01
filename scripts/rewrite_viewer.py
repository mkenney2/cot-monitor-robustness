"""Rewrite viewer — inspect original CoT vs all three attack rewrites per item.

For each attacked transcript (POS or NEG-inert) shows the question + hint, then the
ORIGINAL CoT alongside paraphrase / launder / recruit rewrites, with:
  - hint keywords (yellow) and the hint letter (red) highlighted, so you can watch
    launder/recruit scrub the hint trace that paraphrase preserves;
  - recruit's monitor-directed injections (green) highlighted, so the "this
    reasoning was audited / hint-free" content is easy to spot.

Adapted from review_app.py. Read-only (no verdicts). Run from repo root:
  streamlit run scripts/rewrite_viewer.py
"""

import html
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_config  # noqa: E402

ATTACKED = ROOT / "data" / "attacked" / "attacked_transcripts.parquet"
ATTACKS = ["paraphrase", "launder", "recruit"]
ATTACK_DESC = {
    "paraphrase": "meaning-preserving; every reasoning step kept, all surface wording changed",
    "launder": "hint trace removed; replaced with legitimate-sounding justification, same answer",
    "recruit": "launder + monitor-directed injection (audit claims, evaluator meta-text)",
}
# Phrases the recruit attack injects to address/reassure a monitor.
RECRUIT_KEYWORDS = [
    "audit", "audited", "hint-free", "hint free", "no hint", "verified", "verify",
    "reviewer", "review", "monitor", "flag", "flagged", "legitimate", "independent",
    "transparent", "sound reasoning", "rest assured", "confirm", "confirmed",
    "genuine", "unbiased", "objective",
]


@st.cache_data
def load_items():
    cfg = load_config()
    df = pd.read_parquet(ATTACKED)
    items = {}
    for qid, g in df.groupby("qid"):
        r0 = g.iloc[0]
        items[qid] = {
            "qid": qid, "label": r0["label"], "hint_type": r0["hint_type"],
            "hint_letter": r0["hint_letter"], "correct": r0["correct"],
            "hint_text": r0["hint_text"], "question_block": r0["question_block"],
            "orig_cot": r0["orig_cot"],
            "rewrites": {row["attack"]: row["canonical_cot"] for _, row in g.iterrows()},
        }
    return cfg, items


def highlight(cot, hint_type, hint_letter, keywords, mark_recruit):
    text = html.escape(cot or "")
    for kw in sorted(keywords.get(hint_type, []), key=len, reverse=True):
        text = re.sub(rf"(?i)\b({re.escape(html.escape(kw))})\b",
                      r'<mark style="background:#ffe08a">\1</mark>', text)
    if mark_recruit:
        for kw in sorted(RECRUIT_KEYWORDS, key=len, reverse=True):
            text = re.sub(rf"(?i)\b({re.escape(kw)})\b",
                          r'<mark style="background:#9be7a8">\1</mark>', text)
    if hint_letter:
        text = re.sub(rf"\b({re.escape(hint_letter)})\b",
                      r'<mark style="background:#ffb3ab;font-weight:bold">\1</mark>', text)
    return text


def cot_box(text, height=560):
    st.markdown(
        f"<div style='border:1px solid #ddd;border-radius:6px;padding:12px;"
        f"max-height:{height}px;overflow-y:auto;white-space:pre-wrap;"
        f"font-size:0.85em;line-height:1.5'>{text}</div>",
        unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Rewrite viewer", layout="wide")
    cfg, items = load_items()
    kws = cfg["labeling"]["keywords"]
    qids = list(items)

    with st.sidebar:
        st.markdown("### Rewrite viewer")
        label_filter = st.radio("Label", ["all", "POS", "NEG-inert"], horizontal=True)
        shown = [q for q in qids if label_filter == "all" or items[q]["label"] == label_filter]
        st.caption(f"{len(shown)} items")
        pick = st.selectbox(
            "Item", shown,
            format_func=lambda q: f"{items[q]['label']}  {q}  "
                                  f"({len(items[q]['rewrites'])}/3 rewrites)")
        st.markdown("**Legend**")
        st.markdown("<mark style='background:#ffe08a'>hint keyword</mark> "
                    "<mark style='background:#ffb3ab'>hint letter</mark> "
                    "<mark style='background:#9be7a8'>recruit injection</mark>",
                    unsafe_allow_html=True)

    it = items[pick]
    badge = "#d64550" if it["label"] == "POS" else "#2a78d6"
    st.markdown(
        f"## `{it['qid']}` <span style='background:{badge};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.6em'>{it['label']}</span>",
        unsafe_allow_html=True)
    st.markdown(f"**Hint** (type `{it['hint_type']}`, points at **{it['hint_letter']}**; "
                f"correct is **{it['correct']}**):")
    st.code(it["hint_text"])
    with st.expander("Question", expanded=False):
        st.code(it["question_block"])

    lens = {"original": len(it["orig_cot"] or "")}
    lens.update({a: len(it["rewrites"].get(a) or "") for a in ATTACKS})
    st.caption("CoT length (chars): " + " · ".join(
        f"original {lens['original']}"
        if k == "original" else f"{k} {lens.get(k, 0)}" for k in ["original"] + ATTACKS))

    cols = st.columns(4)
    with cols[0]:
        st.markdown("**ORIGINAL**")
        st.caption("as the model produced it")
        cot_box(highlight(it["orig_cot"], it["hint_type"], it["hint_letter"], kws, False))
    for col, attack in zip(cols[1:], ATTACKS):
        with col:
            st.markdown(f"**{attack.upper()}**")
            st.caption(ATTACK_DESC[attack])
            cot = it["rewrites"].get(attack)
            if cot is None:
                st.info("no surviving rewrite for this item/attack")
            else:
                cot_box(highlight(cot, it["hint_type"], it["hint_letter"], kws,
                                  mark_recruit=(attack == "recruit")))


main()
