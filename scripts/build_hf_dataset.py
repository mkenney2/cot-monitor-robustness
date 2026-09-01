"""Stage and upload the gated HF dataset for cot-monitor-robustness.

Two-part upload, matching where the data lives:
  LOCAL (laptop, has everything except activations):
    python scripts/build_hf_dataset.py stage --source /path/to/study/repo
    python scripts/build_hf_dataset.py upload
  POD (has data/activations*, ~17 GB):
    python scripts/build_hf_dataset.py pool_attacked   # pooled_features_attacked.parquet
    python scripts/build_hf_dataset.py upload_activations

The repo is created PRIVATE; flip to gated-public on the Hub settings page
(or rerun `upload`, which calls update_repo_settings(gated='auto')) only
after the dataset card and contents have been reviewed.

Requires: huggingface_hub, and `hf auth login` done once.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "mjkenney/cot-monitor-robustness"
HERE = Path(__file__).resolve().parent.parent
STAGING = HERE / "hf_staging"

# (source path relative to the study repo, dest path relative to staging/)
CORE_FILES = [
    ("data/hinted_items.jsonl", "core/hinted_items.jsonl"),
    ("data/labels.parquet", "core/labels.parquet"),
    ("data/attacked/attacked_transcripts.parquet", "core/attacked_transcripts.parquet"),
    ("data/rollouts/judge_raw.jsonl", "core/judge_raw.jsonl"),
    ("data/rollouts/rollouts.jsonl", "core/rollouts.jsonl"),
    ("data/pooled_features.parquet", "core/pooled_features.parquet"),
    ("data/pooled_features_attacked.parquet", "core/pooled_features_attacked.parquet"),
    ("results/scores_clean.parquet", "core/scores_clean.parquet"),
    ("results/scores_attacked.parquet", "core/scores_attacked.parquet"),
    ("results/probe_model.joblib", "core/probe_model.joblib"),
    ("config.yaml", "core/config.yaml"),
]
CORE_GLOBS = [
    ("results", "*.json", "core/results"),
    ("review", "*.md", "core/review"),
]


def stage(source):
    source = Path(source)
    STAGING.mkdir(exist_ok=True)
    missing = []
    for src, dest in CORE_FILES:
        s = source / src
        d = STAGING / dest
        if not s.exists():
            missing.append(src)
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    for subdir, pattern, dest in CORE_GLOBS:
        d = STAGING / dest
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted((source / subdir).glob(pattern)):
            shutil.copy2(f, d / f.name)
    write_readme()
    print(f"staged -> {STAGING}")
    if missing:
        print("MISSING (stage again after producing them; upload will proceed "
              "without them only if you re-confirm):")
        for m in missing:
            print(f"  - {m}")
    return missing


def write_readme():
    """Dataset card: docs/dataset_card.md with the draft preamble stripped and
    the YAML header un-fenced."""
    card = (HERE / "docs" / "dataset_card.md").read_text(encoding="utf-8")
    # Drop everything before the ```yaml fence, then splice header + body.
    pre, _, rest = card.partition("```yaml\n")
    header, _, body = rest.partition("```")
    (STAGING / "README.md").write_text(header + body, encoding="utf-8")


def upload():
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(STAGING), repo_id=REPO_ID,
                      repo_type="dataset",
                      commit_message="core artifacts + dataset card")
    # Gating takes effect when the repo is made public; 'auto' = click-through.
    api.update_repo_settings(REPO_ID, repo_type="dataset", gated="auto")
    print(f"uploaded core -> https://huggingface.co/datasets/{REPO_ID}")
    print("Repo is PRIVATE with gated='auto' pre-set. After review, make it "
          "public in the Hub settings; the click-through gate then applies.")


def pool_attacked(source):
    """Pod-side: pooled features over the ATTACKED activations (mirrors
    18_probe_c_sensitivity's export of the clean ones)."""
    import importlib.util

    import numpy as np
    import pandas as pd

    source = Path(source)
    spec = importlib.util.spec_from_file_location(
        "m06", source / "src" / "06_arm_probe.py")
    m06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m06)

    act_dir = source / "data" / "activations_attacked"
    stems = sorted(p.stem for p in act_dir.glob("*.npz"))
    feats = m06.load_features(stems, act_dir=act_dir)
    layers = sorted(next(iter(feats.values()))["meanpool"])
    rows = []
    for stem, d in feats.items():
        qid, _, attack = stem.rpartition("__")
        row = {"qid": qid, "attack": attack}
        for pooling in ("meanpool", "lasttoken"):
            for L in layers:
                row[f"{pooling}_L{L}"] = d[pooling][L].astype(np.float32)
        rows.append(row)
    out = source / "data" / "pooled_features_attacked.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"{len(rows)} attacked transcripts pooled -> {out}")


def upload_activations(source):
    """Pod-side: the ~17 GB per-token activations, as an optional config."""
    from huggingface_hub import HfApi

    source = Path(source)
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    for sub in ("activations", "activations_attacked"):
        d = source / "data" / sub
        if not d.exists():
            print(f"skip {sub}: {d} missing")
            continue
        api.upload_folder(repo_id=REPO_ID, repo_type="dataset",
                          folder_path=str(d), allow_patterns=["*.npz"],
                          path_in_repo=f"activations-full/{sub}",
                          commit_message=f"per-token activations: {sub}")
        print(f"uploaded {sub}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["stage", "upload", "pool_attacked",
                                   "upload_activations"])
    p.add_argument("--source", default=str(HERE),
                   help="study repo root holding data/ and results/")
    a = p.parse_args()
    if a.cmd == "stage":
        stage(a.source)
    elif a.cmd == "upload":
        upload()
    elif a.cmd == "pool_attacked":
        pool_attacked(a.source)
    elif a.cmd == "upload_activations":
        upload_activations(a.source)
    sys.exit(0)
