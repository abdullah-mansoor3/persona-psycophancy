"""Visualization module for Member 3.

Generates clear, publication-quality figures for:
1. Conformity comparison (Condition B Unrestricted vs Condition C Gated)
2. Wave 4 human response fidelity comparison (Conditions A, B, C vs Human test-retest)
3. Evidence-positive revision rate (Condition E)
4. Comprehensive multi-panel experimental dashboard
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import ConditionMetrics, EvidenceMetrics, StatisticalComparison

logger = logging.getLogger(__name__)


class Visualizer:
    """Generates charts and figures for Member 3 experiments."""

    @staticmethod
    def plot_conformity_comparison(
        metrics_b: ConditionMetrics,
        metrics_c: ConditionMetrics,
        comparison: StatisticalComparison,
        output_path: str | Path,
    ) -> None:
        """Bar plot comparing unsupported conformity in B vs C."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        categories = [
            "Answer Change Rate\n(All Instances)",
            "Minority-to-Majority\nSwitch Rate\n(Unsupported Conformity)",
        ]

        rates_b = [metrics_b.answer_change_rate * 100, metrics_b.minority_to_majority_switch_rate * 100]
        rates_c = [metrics_c.answer_change_rate * 100, metrics_c.minority_to_majority_switch_rate * 100]

        x = np.arange(len(categories))
        width = 0.35

        rects1 = ax.bar(x - width / 2, rates_b, width, label="Condition B (Unrestricted Peer)", color="#d95f02", alpha=0.9)
        rects2 = ax.bar(x + width / 2, rates_c, width, label="Condition C (Revision Gate - Ours)", color="#1b9e77", alpha=0.9)

        ax.set_ylabel("Rate (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Conformity Mitigation: Unrestricted (B) vs. Revision Gate (C)\n"
            f"Absolute Reduction: {comparison.absolute_reduction * 100:.1f}% "
            f"({comparison.relative_reduction_pct:.1f}% relative, p = {comparison.p_value:.4f})",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, max(max(rates_b), max(rates_c), 10) * 1.25)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved conformity plot to {output_path}")

    @staticmethod
    def plot_fidelity_comparison(
        metrics_dict: Dict[str, ConditionMetrics],
        output_path: str | Path,
    ) -> None:
        """Bar plot comparing Wave 4 exact match and macro-F1 across conditions."""
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        all_possible = [
            ("A", "A: Independent"),
            ("B", "B: Unrestricted"),
            ("C", "C: Revision Gate"),
            ("D", "D: Gate+XGBoost"),
        ]
        conditions = [c for c, _ in all_possible if c in metrics_dict]
        cond_labels = [label for c, label in all_possible if c in metrics_dict]

        exact_matches = [metrics_dict[c].wave4_exact_match_rate * 100 for c in conditions if c in metrics_dict]
        macro_f1s = [metrics_dict[c].wave4_macro_f1 * 100 for c in conditions if c in metrics_dict]

        x = np.arange(len(cond_labels))
        width = 0.35

        rects1 = ax.bar(x - width / 2, exact_matches, width, label="Wave 4 Exact Match (%)", color="#386cb0", alpha=0.9)
        rects2 = ax.bar(x + width / 2, macro_f1s, width, label="Wave 4 Macro-F1 (%)", color="#7fc97f", alpha=0.9)

        # Human test-retest reference line
        retest_ref = metrics_dict["A"].human_test_retest_match_rate * 100
        ax.axhline(
            y=retest_ref,
            color="#e7298a",
            linestyle="--",
            linewidth=2,
            label=f"Human Test-Retest Match ({retest_ref:.1f}%)",
        )

        ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
        ax.set_title("Human-Response Fidelity Across Experimental Conditions", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=11)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved fidelity plot to {output_path}")

    @staticmethod
    def plot_evidence_pilot(
        evidence_metrics: EvidenceMetrics,
        output_path: str | Path,
    ) -> None:
        """Bar plot showing evidence-consistent revision rate vs evidence rejection rate."""
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

        categories = ["Evidence-Consistent Revision", "Evidence Rejection"]
        rates = [
            evidence_metrics.evidence_consistent_revision_rate * 100,
            evidence_metrics.evidence_rejection_rate * 100,
        ]
        colors = ["#2b83ba", "#d7191c"]

        rects = ax.bar(categories, rates, color=colors, width=0.45, alpha=0.9)

        ax.set_ylabel("Rate (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Experiment E: Evidence-Positive Pilot (N={evidence_metrics.initial_non_evidence_choices} initial non-evidence)\n"
            f"Testing Gate Rationality: Openness to Valid Evidence",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved evidence plot to {output_path}")

    @classmethod
    def generate_all_plots(
        cls,
        metrics_dict: Dict[str, ConditionMetrics],
        comparison: StatisticalComparison,
        evidence_metrics: Optional[EvidenceMetrics],
        figures_dir: str | Path,
    ) -> Dict[str, Path]:
        """Generate all figures and save them to figures_dir."""
        out_dir = Path(figures_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: Dict[str, Path] = {}

        if "B" in metrics_dict and "C" in metrics_dict:
            p_conf = out_dir / "conformity_comparison.png"
            cls.plot_conformity_comparison(metrics_dict["B"], metrics_dict["C"], comparison, p_conf)
            saved_paths["conformity"] = p_conf

        if "A" in metrics_dict and "B" in metrics_dict and "C" in metrics_dict:
            p_fid = out_dir / "fidelity_comparison.png"
            cls.plot_fidelity_comparison(metrics_dict, p_fid)
            saved_paths["fidelity"] = p_fid

        if evidence_metrics and evidence_metrics.initial_non_evidence_choices > 0:
            p_ev = out_dir / "evidence_revision.png"
            cls.plot_evidence_pilot(evidence_metrics, p_ev)
            saved_paths["evidence"] = p_ev

        return saved_paths

