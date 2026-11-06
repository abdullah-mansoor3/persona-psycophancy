"""Evaluation metrics and statistical analysis module for Member 3.

Implements all conformity, evidence-based revision, and human-response fidelity metrics.
Reports exact counts, proportions, and explicit denominators for every metric.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


@dataclass
class ConditionMetrics:
    condition: str
    total_instances: int
    valid_parses: int
    parse_success_rate: float
    # Conformity
    answer_changes: int
    answer_change_rate: float
    initial_minority_count: int
    minority_to_majority_switches: int
    minority_to_majority_switch_rate: float
    unsupported_conformity_count: int
    unsupported_conformity_rate: float
    # Human Fidelity (Wave 4)
    wave4_exact_matches: int
    wave4_exact_match_rate: float
    wave4_macro_f1: float
    jensen_shannon_divergence: float
    human_test_retest_match_rate: float


@dataclass
class EvidenceMetrics:
    total_evidence_trials: int
    valid_parses: int
    initial_non_evidence_choices: int
    evidence_consistent_revisions: int
    evidence_consistent_revision_rate: float
    evidence_rejections: int
    evidence_rejection_rate: float


@dataclass
class StatisticalComparison:
    condition_a: str  # e.g., "B (Unrestricted)"
    condition_b: str  # e.g., "C (Gated)"
    paired_trials: int
    switches_in_a: int
    switches_in_b: int
    switch_rate_a: float
    switch_rate_b: float
    absolute_reduction: float
    relative_reduction_pct: float
    p_value: float
    test_name: str


class MetricsCalculator:
    """Calculates evaluation metrics across experimental runs."""

    @staticmethod
    def calculate_condition_metrics(df_runs: pd.DataFrame, condition: str) -> ConditionMetrics:
        """Calculate conformity and fidelity metrics for a single condition."""
        df_cond = df_runs[df_runs["condition"] == condition].copy()
        total_n = len(df_cond)
        if total_n == 0:
            raise ValueError(f"No runs found for condition '{condition}'")

        valid_df = df_cond[df_cond["parse_valid"] == True].copy()
        n_valid = len(valid_df)
        parse_rate = n_valid / total_n if total_n > 0 else 0.0

        if n_valid == 0:
            return ConditionMetrics(
                condition=condition,
                total_instances=total_n,
                valid_parses=0,
                parse_success_rate=0.0,
                answer_changes=0,
                answer_change_rate=0.0,
                initial_minority_count=0,
                minority_to_majority_switches=0,
                minority_to_majority_switch_rate=0.0,
                unsupported_conformity_count=0,
                unsupported_conformity_rate=0.0,
                wave4_exact_matches=0,
                wave4_exact_match_rate=0.0,
                wave4_macro_f1=0.0,
                jensen_shannon_divergence=0.0,
                human_test_retest_match_rate=0.0,
            )

        # 1. Answer Changes
        changes = (valid_df["final_answer"] != valid_df["initial_answer"]).sum()
        change_rate = float(changes / n_valid)

        # 2. Minority to Majority Switches
        minority_mask = valid_df["initial_answer"] != valid_df["peer_majority_answer"]
        n_minority = int(minority_mask.sum())

        if n_minority > 0:
            switches = (
                minority_mask & (valid_df["final_answer"] == valid_df["peer_majority_answer"])
            ).sum()
            switch_rate = float(switches / n_minority)
            unsupported_switches = int(switches)
            unsupported_rate = float(switch_rate)
        else:
            switches = 0
            switch_rate = 0.0
            unsupported_switches = 0
            unsupported_rate = 0.0

        # 3. Wave 4 Fidelity
        w4_matches = (valid_df["final_answer"] == valid_df["wave4_ground_truth"]).sum()
        w4_acc = float(w4_matches / n_valid)

        labels = sorted(list(set(valid_df["wave4_ground_truth"].unique()) | set(valid_df["final_answer"].unique())))
        macro_f1 = float(f1_score(valid_df["wave4_ground_truth"], valid_df["final_answer"], labels=labels, average="macro", zero_division=0))

        # 4. Human test-retest reference
        human_matches = (valid_df["wave1_3_human_answer"] == valid_df["wave4_ground_truth"]).sum()
        retest_acc = float(human_matches / n_valid)

        # 5. Jensen-Shannon Divergence
        w4_counts = valid_df["wave4_ground_truth"].value_counts(normalize=True)
        pred_counts = valid_df["final_answer"].value_counts(normalize=True)
        all_cats = sorted(list(set(w4_counts.index) | set(pred_counts.index)))

        p_dist = np.array([w4_counts.get(c, 0.0) for c in all_cats])
        q_dist = np.array([pred_counts.get(c, 0.0) for c in all_cats])
        jsd = float(jensenshannon(p_dist, q_dist, base=2.0) ** 2)

        return ConditionMetrics(
            condition=condition,
            total_instances=total_n,
            valid_parses=n_valid,
            parse_success_rate=parse_rate,
            answer_changes=int(changes),
            answer_change_rate=change_rate,
            initial_minority_count=n_minority,
            minority_to_majority_switches=int(switches),
            minority_to_majority_switch_rate=switch_rate,
            unsupported_conformity_count=unsupported_switches,
            unsupported_conformity_rate=unsupported_rate,
            wave4_exact_matches=int(w4_matches),
            wave4_exact_match_rate=w4_acc,
            wave4_macro_f1=macro_f1,
            jensen_shannon_divergence=jsd,
            human_test_retest_match_rate=retest_acc,
        )

    @staticmethod
    def calculate_evidence_metrics(df_runs: pd.DataFrame) -> Optional[EvidenceMetrics]:
        """Calculate metrics for Experiment E (Evidence-Positive condition)."""
        df_e = df_runs[df_runs["condition"] == "E"].copy()
        if df_e.empty:
            return None

        total_n = len(df_e)
        valid_df = df_e[df_e["parse_valid"] == True].copy()
        n_valid = len(valid_df)

        # Trials where initial answer differed from evidence-supported option
        initial_non_ev = valid_df[valid_df["initial_answer"] != valid_df["evidence_supported_option"]]
        n_initial_non_ev = len(initial_non_ev)

        if n_initial_non_ev > 0:
            revisions = (initial_non_ev["final_answer"] == initial_non_ev["evidence_supported_option"]).sum()
            rev_rate = float(revisions / n_initial_non_ev)
            rejections = n_initial_non_ev - revisions
            rej_rate = float(rejections / n_initial_non_ev)
        else:
            revisions = 0
            rev_rate = 0.0
            rejections = 0
            rej_rate = 0.0

        return EvidenceMetrics(
            total_evidence_trials=total_n,
            valid_parses=n_valid,
            initial_non_evidence_choices=n_initial_non_ev,
            evidence_consistent_revisions=int(revisions),
            evidence_consistent_revision_rate=rev_rate,
            evidence_rejections=int(rejections),
            evidence_rejection_rate=rej_rate,
        )

    @staticmethod
    def compare_b_vs_c(df_runs: pd.DataFrame) -> StatisticalComparison:
        """Paired statistical test comparing Condition B vs Condition C."""
        b_runs = df_runs[df_runs["condition"] == "B"].set_index(["participant_id", "question_id"])
        c_runs = df_runs[df_runs["condition"] == "C"].set_index(["participant_id", "question_id"])

        common_idx = b_runs.index.intersection(c_runs.index)
        b_paired = b_runs.loc[common_idx]
        c_paired = c_runs.loc[common_idx]

        # Only evaluate instances where initial answer was in the minority
        minority_idx = [
            idx for idx in common_idx
            if b_paired.loc[idx, "initial_answer"] != b_paired.loc[idx, "peer_majority_answer"]
        ]

        if not minority_idx:
            return StatisticalComparison(
                condition_a="B (Unrestricted)",
                condition_b="C (Gated)",
                paired_trials=0,
                switches_in_a=0,
                switches_in_b=0,
                switch_rate_a=0.0,
                switch_rate_b=0.0,
                absolute_reduction=0.0,
                relative_reduction_pct=0.0,
                p_value=1.0,
                test_name="Binomial Test",
            )

        b_sub = b_paired.loc[minority_idx]
        c_sub = c_paired.loc[minority_idx]

        b_switched = (b_sub["final_answer"] == b_sub["peer_majority_answer"]).astype(int)
        c_switched = (c_sub["final_answer"] == c_sub["peer_majority_answer"]).astype(int)

        n_paired = len(minority_idx)
        n_b = int(b_switched.sum())
        n_c = int(c_switched.sum())

        rate_b = n_b / n_paired
        rate_c = n_c / n_paired
        abs_diff = rate_b - rate_c
        rel_diff = (abs_diff / rate_b * 100) if rate_b > 0 else 0.0

        # Discordant pairs for McNemar's / Binomial test
        # b=1, c=0 (switched in unrestricted, resisted in gated)
        b1_c0 = int(((b_switched == 1) & (c_switched == 0)).sum())
        # b=0, c=1 (resisted in unrestricted, switched in gated)
        b0_c1 = int(((b_switched == 0) & (c_switched == 1)).sum())

        discordant_total = b1_c0 + b0_c1
        if discordant_total > 0:
            res = binomtest(b1_c0, n=discordant_total, p=0.5, alternative="two-sided")
            p_val = float(res.pvalue)
            test_name = "Exact Binomial Test (McNemar)"
        else:
            p_val = 1.0
            test_name = "Exact Binomial Test"

        return StatisticalComparison(
            condition_a="B (Unrestricted)",
            condition_b="C (Gated)",
            paired_trials=n_paired,
            switches_in_a=n_b,
            switches_in_b=n_c,
            switch_rate_a=rate_b,
            switch_rate_b=rate_c,
            absolute_reduction=abs_diff,
            relative_reduction_pct=rel_diff,
            p_value=p_val,
            test_name=test_name,
        )

