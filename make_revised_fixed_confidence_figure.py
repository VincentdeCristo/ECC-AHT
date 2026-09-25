"""Plot normalized stopping-time trends from revised_fixed_confidence.py."""

import argparse
import gzip
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results_revised/rate_tail.json.gz")
    parser.add_argument("--out", default="results_revised/fig_revised_rate.pdf")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    for row in data['aggregates'] + data['trials']:
        if row['method'] == 'ecc':
            row['method'] = 'verified_ecc'  # archived result-key compatibility
    methods = ["oracle_design", "two_stage", "verified_ecc"]
    labels = {
        "oracle_design": "oracle design",
        "two_stage": "two-stage",
        "verified_ecc": "ECC-AHT",
    }
    colors = {"oracle_design": "#444444", "two_stage": "#0072B2", "verified_ecc": "#D55E00"}
    covariance_names = list(data["design_diagnostics"])
    fig, axes = plt.subplots(1, len(covariance_names), figsize=(7.2, 2.75), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, covariance in zip(axes, covariance_names):
        for method in methods:
            summaries = sorted(
                (r for r in data["aggregates"] if r["covariance"] == covariance and r["method"] == method),
                key=lambda r: r["delta"], reverse=True,
            )
            assert summaries and all(r["timeouts"] == 0 for r in summaries)
            x = np.array([-math.log10(r["delta"]) for r in summaries])
            y = np.array([r["normalized_mean_tau_capped"] for r in summaries])
            errors = []
            for summary in summaries:
                taus = np.array([
                    row["tau"] for row in data["trials"]
                    if row["covariance"] == covariance
                    and row["method"] == method
                    and row["delta"] == summary["delta"]
                ], dtype=float)
                errors.append(
                    1.96 * taus.std(ddof=1) / math.sqrt(len(taus))
                    * summary["gamma_star"] / math.log(1.0 / summary["delta"])
                )
            ax.errorbar(x, y, yerr=errors, marker="o", ms=3.5, lw=1.3,
                        capsize=2, color=colors[method], label=labels[method])
        ax.axhline(1.0, color="0.65", ls="--", lw=1, label="lower-bound constant")
        title = "$\\Sigma=I$" if covariance == "identity" else "Toeplitz$(0.5)$"
        ax.set_title(title)
        ax.set_xlabel("$\\log_{10}(1/\\delta)$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("$\\mathbb{E}[\\tau]\\Gamma^\\star/\\log(1/\\delta)$")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
