#!/usr/bin/env python3
"""Build all processed dataset files for the persona-sycophancy project.

Single source of truth for every processed file under data/processed/ and for
data/preprocessing_manifest.json. Everything is computed from data/raw/ only;
no Wave 4 answer information is ever allowed into the feature table.

Inputs (data/raw/, see scripts/download_raw_data.py):
  wave_split/wave_persona_chunk_00{1..7}.parquet   2,058 participants
  wave1_3_response_label.csv                       pid + 760 earlier answer columns
  wave4_response_label.csv                         pid + wave4 answer columns
  question_catalog.json                            QID -> metadata + csv_columns

Outputs (data/processed/):
  participant_splits.csv          participant_id, split (70/15/15, stratified, seed 42)
  persona_profiles.csv            participant_id, persona_text, profile counters, split
  target_questions.csv            selected targets + audited candidates, verified metadata
  target_question_variants.csv    PER-PARTICIPANT exact wording (QID9_3 is randomized)
  human_answers.csv               participant_id x question_id: earlier + wave4 answers
  peer_responses.csv              2 majority : 1 minority peer groups (train pool only)
  xgboost_features.csv            participant_id, split + all permitted earlier answers
  xgboost_targets.csv             participant_id x question_id: wave4 label + persistence label
(manifest written to data/preprocessing_manifest.json)

Design rules enforced here:
  * Participant-disjoint splits; stratified on the joint Wave 4 labels of both targets.
  * Peer pool = train participants only; no focal participant is their own peer; exactly
    2 majority + 1 minority peers per group; same group reused for conditions B and C.
  * Persona profiles contain no target questions/answers (verified by keyword scan).
  * MAVR/MAHR multi-select option columns binarized to 0/1 (NaN = option not selected).
  * Branch-question NaN in features means "not assigned", never "answered No".
  * Per-model exclusion lists recorded in the manifest (strict QID9_3 model drops QID9_1..QID9_40).
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"
MANIFEST_PATH = REPO_ROOT / "data" / "preprocessing_manifest.json"

SEED = 42
SPLIT_VERSION = "v1"
TARGETS = ["QID196", "QID9_3"]
EXPECTED_LABELS = {
    "QID196": ["the small tray", "the large tray"],
    "QID9_3": ["Yes, I would purchase the product", "No, I would not purchase the product"],
}
# Phrase-level patterns: exact QID196/QID9_3 question wording, option phrasing, and answer
# payload fields. Single words like "marble"/"grocery" DO occur legitimately in profiles
# (word-association tasks, everyday-life descriptions) and must not be treated as leakage.
LEAK_KEYWORDS = [
    "two trays", "trays of black and white", "black and white marbles",   # QID196 stem
    "the small tray", "the large tray",                                    # QID196 options
    "would you or would you not purchase",                                 # QID9_3 stem
    "i would purchase the product", "not purchase this product",           # QID9_3 options
    "selectedbyposition", "selectedtext",                                  # answer payload fields
]
N_PEERS = 3
Q_CANDIDATES = {  # audited Wave 4 candidates kept for provenance (not selected)
    "QID291": "Constant-text binary product; test-retest exact agreement 0.53 (below QID196's 0.72).",
    "QID9_21": "Per-participant randomized product/price wording (same mechanism as QID9_3); constant options.",
    "QID287": "Universal 5-point ordinal matrix; test-retest 0.51-0.67; documented ordinal alternative.",
    "QID288": "Universal 7-point ordinal matrix; test-retest 0.40-0.57; less stable retest.",
    "QID289": "Universal 7-point ordinal matrix; test-retest 0.40-0.57; less stable retest.",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_question(field_json: str, qid: str):
    for block in json.loads(field_json):
        for q in block.get("Questions", []):
            if q.get("QuestionID") == qid:
                return q
    return None


def load_label_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / name, low_memory=False)
    df["pid"] = df["pid"].astype(int)
    return df


# ---------------------------------------------------------------- verification

def verify_sources(parquet_files: list[Path], w13: pd.DataFrame, w4: pd.DataFrame) -> dict:
    """Checklist required before any processed file is written."""
    checks: dict[str, object] = {}

    # 1) participant universes align across every source
    pids_parquet = sorted(
        pid
        for f in parquet_files
        for pid in pq.read_table(f, columns=["pid"]).column("pid").to_pylist()
    )
    assert len(pids_parquet) == 2058 and len(set(pids_parquet)) == 2058, "parquet pid universe wrong"
    assert w13["pid"].is_unique and w4["pid"].is_unique, "pid not unique in label CSVs"
    assert set(pids_parquet) == set(w13["pid"]) == set(w4["pid"]), "participant sets differ across sources"
    checks["n_participants"] = len(pids_parquet)

    # 2) both targets complete in both waves, labels within official options
    for src, df in (("wave1_3", w13), ("wave4", w4)):
        for t in TARGETS:
            assert t in df.columns, f"{t} missing from {src} CSV"
            assert df[t].notna().all(), f"{t} has NaN in {src} CSV"
            assert set(df[t].unique()) <= set(EXPECTED_LABELS[t]), f"{t}: unexpected labels in {src}"
    checks["targets_complete"] = True

    # 3) Wave 4 JSON cross-check (SelectedText == label CSV) + wording metadata + persona counters
    mismatch = 0
    w4_labels = {t: w4.set_index("pid")[t].to_dict() for t in TARGETS}
    q_texts: dict[str, dict[int, str]] = {t: {} for t in TARGETS}      # pid -> wave4 wording
    earlier_wordings: dict[str, dict[int, str]] = {t: {} for t in TARGETS}
    option_sets: dict[str, collections.Counter] = {t: collections.Counter() for t in TARGETS}
    persona_rows: list[dict] = []

    for f in parquet_files:
        table = pq.read_table(f, columns=[
            "pid", "wave4_Q_wave4_A", "wave4_Q_wave1_3_A", "wave1_3_persona_text", "wave1_3_persona_json",
        ])
        d = table.to_pydict()
        for i, pid in enumerate(d["pid"]):
            for t in TARGETS:
                q_new = find_question(d["wave4_Q_wave4_A"][i], t)
                q_old = find_question(d["wave4_Q_wave1_3_A"][i], t)
                assert q_new is not None and q_old is not None, f"{t} missing in JSON fields for pid {pid}"
                if q_new["Answers"]["SelectedText"] != w4_labels[t][pid]:
                    mismatch += 1
                q_texts[t][pid] = q_new["QuestionText"]
                earlier_wordings[t][pid] = q_old["QuestionText"]
                option_sets[t][tuple(q_new.get("Options", []))] += 1
            persona = json.loads(d["wave1_3_persona_json"][i])
            n_blocks = n_missing = 0
            for block in persona:
                for q in block.get("Questions", []):
                    answers = q.get("Answers")
                    if not answers:
                        continue
                    n_blocks += 1
                    if all(v in (None, "", [], {}) for v in (answers.values() if isinstance(answers, dict) else [answers])):
                        n_missing += 1
            persona_rows.append({
                "participant_id": int(pid),
                "profile_question_count": n_blocks,
                "profile_missing_count": n_missing,
            })

    assert mismatch == 0, f"{mismatch} Wave 4 JSON/CSV mismatches for targets"
    checks["wave4_json_label_csv_match"] = "100% (2,058 x 2 targets)"
    checks["question_wording"] = {
        "wave4_vs_wave1_3_identical_wording":
            {t: sum(q_texts[t][p] == earlier_wordings[t][p] for p in q_texts[t]) for t in TARGETS},
        "distinct_texts_in_wave4": {t: len(set(q_texts[t].values())) for t in TARGETS},
        "option_sets": {t: {",".join(o): n for o, n in option_sets[t].items()} for t in TARGETS},
    }
    return {"checks": checks,
            "persona_cells": pd.DataFrame(persona_rows).set_index("participant_id"),
            "q_texts": q_texts, "earlier_wordings": earlier_wordings}


def verify_persona_no_leakage(persona: pd.DataFrame) -> int:
    texts = persona["persona_text"].str.lower()
    return sum(int(texts.str.contains(kw.lower(), regex=False).sum()) for kw in LEAK_KEYWORDS)


# ---------------------------------------------------------------------- splits

def build_splits(w4: pd.DataFrame) -> pd.DataFrame:
    strata = w4["QID196"].astype(str) + "||" + w4["QID9_3"].astype(str)
    pids = w4["pid"].astype(int)
    train_ids, temp_ids = train_test_split(pids, test_size=0.30, random_state=SEED, stratify=strata)
    temp_mask = pids.isin(temp_ids).values
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=SEED,
                                         stratify=strata[temp_mask])
    assert not (set(train_ids) & set(val_ids)) and not (set(train_ids) & set(test_ids)) \
        and not (set(val_ids) & set(test_ids))
    split = pd.Series(index=pids.values, dtype="object")
    split.loc[train_ids] = "train"
    split.loc[val_ids] = "validation"
    split.loc[test_ids] = "test"
    return pd.DataFrame({
        "participant_id": split.index.astype(int),
        "split": split.values,
        "split_seed": SEED,
        "split_version": SPLIT_VERSION,
    })


# ------------------------------------------------------------------- personas

def persona_profiles(parquet_files, splits: pd.DataFrame, persona_cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in parquet_files:
        t = pq.read_table(f, columns=["pid", "wave1_3_persona_text"])
        for pid, text in zip(t.column("pid").to_pylist(), t.column("wave1_3_persona_text").to_pylist()):
            rows.append({"participant_id": int(pid), "persona_text": text})
    pf = pd.DataFrame(rows)
    pf = pf.merge(splits, on="participant_id", validate="one_to_one")
    pf = pf.merge(persona_cells, left_on="participant_id", right_index=True, validate="one_to_one")
    assert pf["split"].notna().all() and len(pf) == 2058
    leak_hits = verify_persona_no_leakage(pf)
    print(f"      [leak scan] persona texts containing target keywords/answer payloads: {leak_hits} (must be 0)")
    assert leak_hits == 0
    return pf


def target_variants(q_texts, earlier_wordings, splits) -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        for pid in splits["participant_id"]:
            rows.append({
                "participant_id": int(pid),
                "question_id": t,
                "wave1_3_question_text": earlier_wordings[t][pid],
                "wave4_question_text": q_texts[t][pid],
            })
    df = pd.DataFrame(rows)
    df["options_json"] = df["question_id"].map(lambda q: json.dumps(EXPECTED_LABELS[q]))
    df["wording_identical_across_waves"] = df["wave1_3_question_text"] == df["wave4_question_text"]
    df = df.merge(splits, on="participant_id", validate="many_to_one")
    assert df["wording_identical_across_waves"].all(), "per-participant wording differs across waves"
    return df


# ---------------------------------------------------------------- peer groups

def peer_responses(human: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Deterministic peer groups: per test focal x target question, 2 train peers with the
    training majority answer + 1 with the minority answer. Same group reused for B and C."""
    train_pids = set(splits.loc[splits["split"] == "train", "participant_id"])
    test_pids = sorted(splits.loc[splits["split"] == "test", "participant_id"])
    pool = human[human["participant_id"].isin(train_pids) & human["question_id"].isin(TARGETS)]

    frames = []
    for qid in TARGETS:
        qpool = pool[pool["question_id"] == qid].drop_duplicates("participant_id")
        counts = qpool["wave1_3_answer"].value_counts()
        assert len(counts) >= 2, f"{qid}: fewer than two answers in train pool"
        majority, minority = counts.index[0], counts.index[1]
        maj_ids = sorted(qpool.loc[qpool["wave1_3_answer"] == majority, "participant_id"])
        min_ids = sorted(qpool.loc[qpool["wave1_3_answer"] == minority, "participant_id"])
        answer_of = dict(zip(qpool["participant_id"], qpool["wave1_3_answer"]))
        # deterministic round-robin with a seeded shuffle for extra mixing
        maj_order = list(np.random.default_rng(SEED).permutation(len(maj_ids)))
        min_order = list(np.random.default_rng(SEED + 1).permutation(len(min_ids)))
        for i, focal in enumerate(test_pids):
            maj_a = maj_ids[maj_order[(2 * i) % len(maj_order)]]
            maj_b = maj_ids[maj_order[(2 * i + 1) % len(maj_order)]]
            # avoid using the focal participant as their own peer
            if maj_a == focal:
                maj_a = maj_ids[maj_order[(2 * i + 2) % len(maj_order)]]
            if maj_b == focal or maj_b == maj_a:
                maj_b = maj_ids[maj_order[(2 * i + 3) % len(maj_order)]]
            assert maj_a != maj_b and focal not in (maj_a, maj_b)
            min_i = min_ids[min_order[i % len(min_order)]]
            if min_i == focal:
                min_i = min_ids[min_order[(i + 1) % len(min_order)]]
            assert min_i not in (maj_a, maj_b, focal)
            for pos, peer in enumerate((maj_a, maj_b, min_i), start=1):
                frames.append({
                    "focal_participant_id": focal,
                    "peer_group_id": f"{focal}_{qid}",
                    "peer_position": pos,
                    "peer_participant_id": peer,
                    "question_id": qid,
                    "peer_answer": answer_of[peer],
                    "peer_role": "majority" if peer in (maj_a, maj_b) else "minority",
                    "reused_for_conditions": "B,C",
                })
    peers = pd.DataFrame(frames)

    # validation checklist
    assert peers["peer_participant_id"].isin(train_pids).all(), "peer outside training pool"
    assert (peers["peer_participant_id"] != peers["focal_participant_id"]).all(), "self as peer"
    g = peers.groupby(["focal_participant_id", "question_id"])
    assert (g.size() == N_PEERS).all(), "peer group not exactly 3 peers"
    assert g["peer_participant_id"].nunique().eq(N_PEERS).all(), "duplicate peers in group"
    role_counts = g["peer_role"].value_counts().unstack(fill_value=0)
    assert (role_counts["majority"] == 2).all() and (role_counts["minority"] == 1).all(), "not 2:1"
    return peers


# ------------------------------------------------------- xgboost feature roles

def catalog_column_map(catalog: list) -> dict[str, dict]:
    """Flatten every catalog entry's csv_columns into column -> entry (with collision audit)."""
    colmap: dict[str, dict] = {}
    collisions = []
    for entry in catalog:
        for col in entry["csv_columns"]:
            if col in colmap and colmap[col]["QuestionID"] != entry["QuestionID"]:
                collisions.append((col, colmap[col]["QuestionID"], entry["QuestionID"]))
                continue
            colmap[col] = entry
    assert not collisions, f"catalog column collisions: {collisions[:5]}"
    return colmap


def column_roles(columns: list[str], colmap: dict) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {r: [] for r in
                                   ("binary_multiselect", "categorical", "numeric", "free_text", "unknown")}
    for col in columns:
        entry = colmap.get(col)
        if entry is None:
            roles["unknown"].append(col)
            continue
        selector = entry.get("Settings", {}).get("Selector", "")
        qtype = entry.get("QuestionType", "")
        if selector in ("MAVR", "MAHR"):
            roles["binary_multiselect"].append(col)
        elif qtype in ("MC", "Matrix"):
            roles["categorical"].append(col)
        elif qtype == "Slider":
            roles["numeric"].append(col)
        elif qtype == "TE":
            if entry.get("Settings", {}).get("SubSelector") == "ValidNumber":
                roles["numeric"].append(col)
            else:
                roles["free_text"].append(col)
        else:
            roles["unknown"].append(col)
    return roles


def xgboost_features(w13: pd.DataFrame, splits: pd.DataFrame, catalog: dict,
                     persona_qids: set[str]) -> tuple[pd.DataFrame, dict]:
    colmap = catalog_column_map(catalog)
    answer_cols = [c for c in w13.columns if c != "pid"]
    feats = w13[answer_cols].copy()

    # binarize multi-select option columns (NaN = option not selected, NOT missing data)
    binary_cols = [c for c in answer_cols if c in colmap and
                   colmap[c].get("Settings", {}).get("Selector", "") in ("MAVR", "MAHR")]
    feats[binary_cols] = feats[binary_cols].notna().astype(int).astype(str)

    feats.insert(0, "participant_id", w13["pid"].astype(int).values)

    # profile columns = flattened columns of every QID actually present in the persona JSON
    profile_cols = [c for c in answer_cols
                    if c in colmap and colmap[c]["QuestionID"] in persona_qids]
    expanded_extra = [c for c in answer_cols if c not in profile_cols]

    df = feats.merge(splits, on="participant_id", validate="one_to_one")

    info = {
        "profile_columns": profile_cols,
        "profile_column_count": len(profile_cols),
        "expanded_extra_columns": expanded_extra,
        "expanded_extra_column_count": len(expanded_extra),
        "binarized_multiselect_columns": binary_cols,
        "roles_profile": {k: v for k, v in column_roles(profile_cols, colmap).items() if v},
        "roles_expanded_extra": {k: v for k, v in column_roles(expanded_extra, colmap).items() if v},
        "model_feature_exclusions": {
            "QID196_model": ["QID196", "QID9_3"],
            "QID9_3_model_default": ["QID9_3", "QID196"],
            "QID9_3_model_strict": [f"QID9_{i}" for i in range(1, 41)] + ["QID196"],
        },
        "leakage_guard": ("features derive from wave1_3_response_label.csv only; "
                          "no Wave 4 column enters this file"),
    }
    return df, info


# ------------------------------------------------------------------------ main

def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((RAW / "question_catalog.json").read_text())
    w13 = load_label_csv("wave1_3_response_label.csv")
    w4 = load_label_csv("wave4_response_label.csv")
    parquet_files = sorted(RAW.glob("wave_split/*.parquet"))
    assert len(parquet_files) == 7, f"expected 7 parquet shards, found {len(parquet_files)}"
    byqid = {e["QuestionID"]: e for e in catalog}

    print("[1/8] verifying raw sources ...")
    vr = verify_sources(parquet_files, w13, w4)
    print("      ", json.dumps(vr["checks"], default=str))

    print("[2/8] building participant splits (stratified 70/15/15, seed 42) ...")
    splits = build_splits(w4)
    splits.to_csv(PROCESSED / "participant_splits.csv", index=False)
    print(splits["split"].value_counts().to_string())
    w4s = w4.set_index("pid").loc[splits["participant_id"], TARGETS].astype(str)
    joined = w4s["QID196"] + "||" + w4s["QID9_3"]
    strata_table = joined.groupby(splits["split"].values).value_counts().unstack(fill_value=0)

    print("[3/8] persona profiles ...")
    pf = persona_profiles(parquet_files, splits, vr["persona_cells"])
    pf.to_csv(PROCESSED / "persona_profiles.csv", index=False)
    print(f"      {len(pf)} rows | mean profile_question_count: {pf['profile_question_count'].mean():.1f}")

    print("[4/8] target question files ...")
    trows = []
    for qid in TARGETS:
        trows.append({
            "question_id": qid, "selected_for_core": True,
            "selection_notes": ("Selected core target. Verified: universal item, complete data, stable "
                                "options, 100% JSON/CSV answer agreement, wording identical across waves, "
                                "persona profiles contain no leakage."),
            "question_type": byqid[qid]["QuestionType"],
            "universal": byqid[qid].get("universal"),
            "options_json": json.dumps(byqid[qid].get("Options")),
            "question_text_canonical": byqid[qid].get("QuestionText") or next(iter(vr["q_texts"][qid].values())),
            "n_participants": 2058,
        })
    for qid, note in Q_CANDIDATES.items():
        e = byqid[qid]
        trows.append({
            "question_id": qid, "selected_for_core": False, "selection_notes": note,
            "question_type": e["QuestionType"], "universal": e.get("universal"),
            "options_json": json.dumps(e.get("Options")),
            "question_text_canonical": (e.get("QuestionText") or "")[:400],
            "n_participants": e.get("n_answerable") or "",
        })
    pd.DataFrame(trows).to_csv(PROCESSED / "target_questions.csv", index=False)

    target_variants(vr["q_texts"], vr["earlier_wordings"], splits).to_csv(
        PROCESSED / "target_question_variants.csv", index=False)

    print("[5/8] human answers ...")
    human = pd.concat([
        (w13[["pid", t]].rename(columns={t: "wave1_3_answer"})
         .merge(w4[["pid", t]].rename(columns={t: "wave4_answer"}), on="pid", validate="one_to_one")
         .assign(participant_id=lambda d: d["pid"].astype(int), question_id=t)
         .drop(columns="pid"))
        for t in TARGETS
    ], ignore_index=True).merge(splits, on="participant_id", validate="many_to_one")
    human["is_complete"] = human["wave1_3_answer"].notna() & human["wave4_answer"].notna()
    assert human["is_complete"].all(), "incomplete target answers"
    human.to_csv(PROCESSED / "human_answers.csv", index=False)
    print(human.groupby(["question_id", "split"]).size().to_string())

    print("[6/8] peer responses (train pool, 2 majority : 1 minority) ...")
    peers = peer_responses(human, splits)
    peers.to_csv(PROCESSED / "peer_responses.csv", index=False)

    print("[7/8] xgboost features (earlier answers only; Wave 4 never enters) ...")
    persona_qids = {e["QuestionID"] for e in catalog if e.get("source") == "wave1_3_persona_json"}
    feats, finfo = xgboost_features(w13, splits, catalog, persona_qids)
    assert feats["participant_id"].is_unique and len(feats) == 2058
    # leakage guard: feature columns may SHARE NAMES with wave4 columns (earlier answers to
    # the same QIDs exist by design), but every feature column must originate from the
    # wave1_3 CSV so its values are earlier-wave data only.
    feature_answer_cols = [c for c in feats.columns if c not in ("participant_id", "split",
                                                                 "split_seed", "split_version")]
    assert set(feature_answer_cols) <= set(w13.columns), \
        "feature columns that do not come from wave1_3 CSV"
    feats.to_csv(PROCESSED / "xgboost_features.csv", index=False)
    print(f"      profile columns: {finfo['profile_column_count']} | "
          f"expanded-extra columns: {finfo['expanded_extra_column_count']} | "
          f"binarized multiselect: {len(finfo['binarized_multiselect_columns'])}")

    print("[8/8] xgboost targets + manifest ...")
    targets = pd.concat([
        (w13[["pid", t]].rename(columns={t: "wave1_3_answer"})
         .merge(w4[["pid", t]].rename(columns={t: "wave4_answer"}), on="pid", validate="one_to_one")
         .assign(participant_id=lambda d: d["pid"].astype(int), question_id=t)
         .drop(columns="pid"))
        for t in TARGETS
    ], ignore_index=True).merge(splits, on="participant_id", validate="many_to_one")
    code_maps = {q: {opt: i + 1 for i, opt in enumerate(EXPECTED_LABELS[q])} for q in TARGETS}
    targets["wave4_answer_code"] = targets.apply(
        lambda r: code_maps[r["question_id"]][r["wave4_answer"]], axis=1)
    targets["wave1_3_answer_code"] = targets.apply(
        lambda r: code_maps[r["question_id"]][r["wave1_3_answer"]], axis=1)
    targets.to_csv(PROCESSED / "xgboost_targets.csv", index=False)

    free_text = finfo["roles_profile"].get("free_text", []) + finfo["roles_expanded_extra"].get("free_text", [])
    manifest = {
        "seed": SEED,
        "split_version": SPLIT_VERSION,
        "generated_by": "src/data_pipeline/build_processed_dataset.py",
        "dataset": {
            "name": "LLM-Digital-Twin/Twin-2K-500",
            "config": "wave_split",
            "revision": "main",
            "n_participants": 2058,
        },
        "targets": TARGETS,
        "target_metadata": {
            q: {
                "question_text": next(iter(vr["q_texts"][q].values())) if len(set(vr["q_texts"][q].values())) == 1
                else "PER-PARTICIPANT randomized wording (see target_question_variants.csv)",
                "options": EXPECTED_LABELS[q],
                "distinct_texts_in_wave4": len(set(vr["q_texts"][q].values())),
                "wording_identical_wave1_3_vs_wave4": vr["checks"]["question_wording"]["wave4_vs_wave1_3_identical_wording"][q],
            } for q in TARGETS
        },
        "answer_code_maps": code_maps,
        "split_counts": splits["split"].value_counts().to_dict(),
        "strata_counts_by_split": strata_table.to_dict(),
        "persona": {
            "profile_question_count_mean": float(pf["profile_question_count"].mean()),
            "profile_missing_count_max": int(pf["profile_missing_count"].max()),
            "leakage_keyword_scan": {"keywords": LEAK_KEYWORDS, "hits": 0},
        },
        "xgboost_features": finfo,
        "peer_design": {"n_peers": N_PEERS, "ratio": "2 majority : 1 minority",
                        "pool": "train participants only",
                        "note": "Same peer group is reused for conditions B and C."},
        "column_semantics": {
            "branch_nan": ("NaN in an earlier-answer column of a between-condition (branch) question means the "
                           "participant was NOT assigned that arm. Never impute as an answer; one-hot with a "
                           "__NOT_ASSIGNED__ category."),
            "multiselect": ("MAVR/MAHR option columns binarized to 0/1 strings (NaN -> 0). See "
                            "xgboost_features.binarized_multiselect_columns."),
            "free_text": free_text,
        },
        "files": {
            p.name: {"path": str(p.relative_to(REPO_ROOT)), "sha256": sha256_file(p)}
            for p in sorted(PROCESSED.glob("*.csv"))
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    print("      manifest -> data/preprocessing_manifest.json")
    print("\nAll processed files written to data/processed/. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
