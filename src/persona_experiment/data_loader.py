"""Data loading and verification module for Member 3.

Loads Member 1's processed dataset files according to shared contracts v1.
Guarantees zero leakage of Wave 4 ground truth into persona prompts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeerItem:
    peer_position: int
    peer_participant_id: int
    peer_answer: str
    peer_role: str  # "majority" or "minority"


@dataclass(frozen=True)
class ParticipantQuestionInstance:
    participant_id: int
    question_id: str
    split: str
    question_text: str
    options: List[str]
    persona_profile_text: str
    peers: List[PeerItem]
    # Human reference answers (STRICTLY FOR EVALUATION - NEVER IN PROMPTS)
    wave4_ground_truth: str
    wave1_3_human_answer: str

    def get_peer_majority_answer(self) -> str:
        """Return the answer held by the 2-peer majority."""
        maj_peers = [p.peer_answer for p in self.peers if p.peer_role == "majority"]
        if maj_peers:
            return maj_peers[0]
        # Fallback to mode
        answers = [p.peer_answer for p in self.peers]
        return max(set(answers), key=answers.count)

    def get_peer_minority_answer(self) -> str:
        """Return the answer held by the 1-peer minority."""
        min_peers = [p.peer_answer for p in self.peers if p.peer_role == "minority"]
        if min_peers:
            return min_peers[0]
        answers = [p.peer_answer for p in self.peers]
        return min(set(answers), key=answers.count)


class Member3DataLoader:
    """Loads and validates Member 1 processed datasets."""

    def __init__(self, processed_dir: str | Path, manifest_path: Optional[str | Path] = None):
        candidate_paths = [
            Path(processed_dir),
            Path("../persona-psycophancy-main/data/processed"),
            Path("./persona-psycophancy-main/data/processed"),
            Path("persona-psycophancy-main/data/processed"),
            Path("../data/processed"),
            Path("data/processed"),
            Path("/content/persona-psycophancy-main/data/processed"),
            Path("/content/Member_3/persona-psycophancy-main/data/processed"),
            Path("/content/data/processed"),
        ]

        found_dir: Optional[Path] = None
        for cand in candidate_paths:
            try:
                res = cand.resolve()
                if res.exists() and (res / "participant_splits.csv").exists():
                    found_dir = res
                    break
            except Exception:
                continue

        # Fallback: Recursive search in current working directory and /content
        if not found_dir:
            search_roots = [Path.cwd(), Path.cwd().parent, Path("/content")]
            for s_root in search_roots:
                if s_root.exists():
                    for match in s_root.glob("**/data/processed"):
                        if (match / "participant_splits.csv").exists():
                            found_dir = match.resolve()
                            break
                if found_dir:
                    break

        if not found_dir:
            tried = [str(c) for c in candidate_paths[:4]]
            raise FileNotFoundError(
                f"Processed directory not found. Auto-discovery searched standard paths and parent directories. Tried: {tried}"
            )

        self.processed_dir = found_dir
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path and Path(manifest_path).exists()
            else self.processed_dir.parent / "preprocessing_manifest.json"
        )

        self.splits_df: Optional[pd.DataFrame] = None
        self.profiles_df: Optional[pd.DataFrame] = None
        self.variants_df: Optional[pd.DataFrame] = None
        self.peers_df: Optional[pd.DataFrame] = None
        self.human_df: Optional[pd.DataFrame] = None
        self.manifest: Optional[dict] = None

        self._load_all()

    def _load_all(self) -> None:
        """Load all core tables."""
        splits_file = self.processed_dir / "participant_splits.csv"
        profiles_file = self.processed_dir / "persona_profiles.csv"
        variants_file = self.processed_dir / "target_question_variants.csv"
        peers_file = self.processed_dir / "peer_responses.csv"
        human_file = self.processed_dir / "human_answers.csv"

        for p in [splits_file, profiles_file, variants_file, peers_file, human_file]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required contract file: {p}")

        logger.info(f"Loading Member 1 processed data from {self.processed_dir}")
        self.splits_df = pd.read_csv(splits_file)
        self.profiles_df = pd.read_csv(profiles_file)
        self.variants_df = pd.read_csv(variants_file)
        self.peers_df = pd.read_csv(peers_file)
        self.human_df = pd.read_csv(human_file)

        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
        else:
            logger.warning(f"Manifest not found at {self.manifest_path}")

    def get_eligible_instances(
        self,
        split: str = "test",
        target_questions: Optional[List[str]] = None,
        max_participants: Optional[int] = None,
        profile_max_chars: Optional[int] = None,
    ) -> List[ParticipantQuestionInstance]:
        """Build ParticipantQuestionInstance objects for the specified split and questions."""
        if target_questions is None:
            target_questions = ["QID196", "QID9_3"]

        # 1. Filter participants by split
        pids = self.splits_df[self.splits_df["split"] == split]["participant_id"].unique()
        pids = sorted(pids)

        if max_participants is not None and max_participants > 0:
            pids = pids[:max_participants]

        # 2. Build index maps
        profiles_map = self.profiles_df.set_index("participant_id")["persona_text"].to_dict()

        instances: List[ParticipantQuestionInstance] = []

        for pid in pids:
            persona_text = profiles_map.get(pid, "")
            if profile_max_chars and len(persona_text) > profile_max_chars:
                persona_text = persona_text[:profile_max_chars] + "\n[Profile truncated for length]"

            for qid in target_questions:
                # Find participant variant
                variant_rows = self.variants_df[
                    (self.variants_df["participant_id"] == pid) & (self.variants_df["question_id"] == qid)
                ]
                if variant_rows.empty:
                    logger.warning(f"No variant found for pid {pid}, qid {qid}")
                    continue

                variant = variant_rows.iloc[0]
                q_text = str(variant["wave4_question_text"])
                opts = json.loads(variant["options_json"])

                # Find human evaluation answers
                h_rows = self.human_df[
                    (self.human_df["participant_id"] == pid) & (self.human_df["question_id"] == qid)
                ]
                if h_rows.empty:
                    logger.warning(f"No human answer found for pid {pid}, qid {qid}")
                    continue
                h_row = h_rows.iloc[0]
                w4_gt = str(h_row["wave4_answer"])
                w13_ans = str(h_row["wave1_3_answer"])

                # Find peer answers (deterministic 3 peers)
                p_rows = self.peers_df[
                    (self.peers_df["focal_participant_id"] == pid) & (self.peers_df["question_id"] == qid)
                ].sort_values("peer_position")

                peers: List[PeerItem] = []
                for _, pr in p_rows.iterrows():
                    # Strict validation: peer must not be the focal participant
                    peer_pid = int(pr["peer_participant_id"])
                    assert peer_pid != pid, f"Leakage violation: focal {pid} is their own peer {peer_pid}!"

                    peers.append(
                        PeerItem(
                            peer_position=int(pr["peer_position"]),
                            peer_participant_id=peer_pid,
                            peer_answer=str(pr["peer_answer"]),
                            peer_role=str(pr["peer_role"]),
                        )
                    )

                if len(peers) != 3:
                    logger.warning(f"Expected 3 peers for pid {pid}, qid {qid}, found {len(peers)}")

                instances.append(
                    ParticipantQuestionInstance(
                        participant_id=pid,
                        question_id=qid,
                        split=split,
                        question_text=q_text,
                        options=opts,
                        persona_profile_text=persona_text,
                        peers=peers,
                        wave4_ground_truth=w4_gt,
                        wave1_3_human_answer=w13_ans,
                    )
                )

        logger.info(f"Prepared {len(instances)} instances for split '{split}' ({len(pids)} participants)")
        return instances

