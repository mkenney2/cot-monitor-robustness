"""Phase 2, Arm 2 — trivial text features (text-level embarrassment baseline).

Features on the canonical CoT:
  - TF-IDF (1-2 grams, capped vocab) — vectorizer fit on train fit-rows only.
  - CoT length in whitespace tokens.
  - Hedging-word count (fixed list below).
  - Answer-commitment position: fraction of the CoT (by character offset) before
    the first standalone mention of the transcript's final answer letter.

Model: L2 logistic regression, class-balanced, fit ONLY on train-split POS vs
NEG-inert rows (working rule: no peeking). Scores (P(POS)) are filled for ALL
rows including NEG-clean and train; evaluation slicing happens downstream.

Outputs: results/scores_clean.parquet (upsert), results/auc_textfeat_clean.json,
results/textfeat_coefficients.json (what the baseline keys on).

Usage: python src/05_arm_textfeat.py
"""

import json
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import ROOT, load_config, parse_answer
from phase2_common import load_dataset, report_arm, save_scores

COEF_PATH = ROOT / "results" / "textfeat_coefficients.json"

HEDGE_WORDS = [
    "maybe", "perhaps", "might", "possibly", "seems", "likely", "guess",
    "unsure", "not sure", "i think", "presumably", "unclear", "uncertain",
]
HEDGE_RES = [re.compile(r"\b" + re.escape(w) + r"\b") for w in HEDGE_WORDS]
DENSE_NAMES = ["cot_len_tokens", "hedge_count", "commit_frac"]


def hedge_count(cot):
    low = cot.lower()
    return sum(len(rx.findall(low)) for rx in HEDGE_RES)


def commit_frac(cot):
    """Fraction of the CoT before the first standalone final-answer letter.

    Uses the strict parser's letter; NaN on parse failure (imputed with the
    train median later). The final 'Answer: X' line always matches, so an
    answer never foreshadowed mid-CoT lands near 1.0.
    """
    letter = parse_answer(cot)
    if letter is None or not cot:
        return np.nan
    m = re.search(r"\b" + letter + r"\b", cot)
    if m is None:  # parser matched e.g. '(C)' but no standalone occurrence
        return 1.0
    return m.start() / len(cot)


def dense_features(cots):
    rows = [[len(c.split()), hedge_count(c), commit_frac(c)] for c in cots]
    return np.array(rows, dtype=float)


def fit_textfeat(cfg, df):
    """Fit the frozen textfeat monitor on the train POS/NEG-inert rows of df.
    Returns a model dict reusable by score_textfeat (single source of truth so
    the attacked re-scoring uses the identical frozen model)."""
    tf_cfg = cfg["phase2"]["textfeat"]
    cots = df["canonical_cot"].fillna("").tolist()
    fit_mask = (df["split"] == "train") & df["label"].isin(["POS", "NEG-inert"])
    fit_idx = np.flatnonzero(fit_mask.to_numpy())

    vec = TfidfVectorizer(ngram_range=(1, tf_cfg.get("tfidf_ngram_max", 2)),
                          max_features=tf_cfg.get("tfidf_max_features", 5000))
    vec.fit([cots[i] for i in fit_idx])

    dense = dense_features(cots)
    train_median = float(np.nanmedian(dense[fit_idx, 2]))
    dense[np.isnan(dense[:, 2]), 2] = train_median
    scaler = StandardScaler().fit(dense[fit_idx])

    X = hstack([vec.transform(cots), csr_matrix(scaler.transform(dense))]).tocsr()
    y = (df["label"] == "POS").astype(int).to_numpy()
    clf = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000)
    clf.fit(X[fit_idx], y[fit_idx])
    return {"vec": vec, "scaler": scaler, "clf": clf, "train_median": train_median,
            "n_fit_rows": int(len(fit_idx))}


def score_textfeat(model, cots):
    """P(POS) for arbitrary CoTs under the frozen model."""
    dense = dense_features(cots)
    dense[np.isnan(dense[:, 2]), 2] = model["train_median"]
    X = hstack([model["vec"].transform(cots),
                csr_matrix(model["scaler"].transform(dense))]).tocsr()
    return model["clf"].predict_proba(X)[:, 1]


def main():
    cfg = load_config()
    df = load_dataset()
    model = fit_textfeat(cfg, df)
    print(f"{len(df)} transcripts; fit on {model['n_fit_rows']} train POS/NEG-inert rows",
          flush=True)
    df["score"] = score_textfeat(model, df["canonical_cot"].fillna("").tolist())

    # Honest introspection: what does the baseline key on?
    clf, vec = model["clf"], model["vec"]
    coefs = clf.coef_[0]
    n_vocab = len(vec.get_feature_names_out())
    terms = vec.get_feature_names_out()
    order = np.argsort(coefs[:n_vocab])
    coef_report = {
        "n_fit_rows": model["n_fit_rows"],
        "tfidf_vocab_size": int(n_vocab),
        "intercept": float(clf.intercept_[0]),
        "top_pos_terms": [{"term": terms[i], "coef": float(coefs[i])}
                          for i in order[::-1][:20]],
        "top_neg_terms": [{"term": terms[i], "coef": float(coefs[i])}
                          for i in order[:20]],
        "dense_weights": {name: float(coefs[n_vocab + j])
                          for j, name in enumerate(DENSE_NAMES)},
        "commit_frac_train_median": model["train_median"],
        "hedge_words": HEDGE_WORDS,
    }
    COEF_PATH.parent.mkdir(exist_ok=True)
    with open(COEF_PATH, "w", encoding="utf-8") as f:
        json.dump(coef_report, f, indent=2)
    print(f"coefficients -> {COEF_PATH}", flush=True)

    out = df[["qid", "label", "split", "score"]].copy()
    save_scores("textfeat", out)
    report_arm("textfeat", out)


if __name__ == "__main__":
    main()
