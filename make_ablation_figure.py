"""
Second figure for the Experiments section: the ablation, and two theorem checks.

Rendered at 7.0in (double-column text width ~7.16in) with an 8pt base font, so
the glyphs are not downscaled -- a defect of the original figures.

Data provenance: the ablation and budget rows come from ablation_fc.py and
exp_rate_validation.py; the signal rows from exp_rate_validation.py. Empirical error
was 0.000 in every configuration except where the caption says otherwise.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---- (a) ablation, K=20 n=2 B=2 delta=0.05, 5 seeds ----------------------
RULES = ["ECC-AHT", "SimpleDiff", "Diagonal", "CostFree", "RSP", "CombGapE", "TTTS"]
ISO   = [87.8, 87.8, 87.8, 87.8, 311.2, 213.8, 144.8]      # Sigma = I
TOEP  = [251.6, 82.2, 86.2, 187.2, 372.4, 213.8, 287.8]     # Toeplitz(0.5)

# ---- (b) budget sweep, identity, delta=0.05 ------------------------------
BUD_B = [1.0, 2.0, 4.0]
BUD_T = [116.7, 116.7, 116.7]

# ---- (c) signal sweep, identity, delta=0.05 -----------------------------
SIG_D  = [0.5, 1.0, 2.0]
SIG_G  = [0.0694, 0.2778, 1.1111]                           # Gamma*, exact SDP
SIG_T  = [557.5, 116.7, 33.9]

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))

x = np.arange(len(RULES)); w = 0.38
ax[0].bar(x - w/2, ISO, w, label=r"$\Sigma=I$", color="#1f4e79")
ax[0].bar(x + w/2, TOEP, w, label=r"Toeplitz$(0.5)$", color="#c00000")
ax[0].axhline(87.8, ls=":", lw=0.8, color="0.4")
ax[0].set_xticks(x); ax[0].set_xticklabels(RULES, rotation=38, ha="right")
ax[0].set_ylabel(r"$\mathbb{E}[\tau]$")
ax[0].set_title("(a) the design rule decides")
ax[0].legend(frameon=False, loc="upper left")

ax[1].plot(BUD_B, BUD_T, "o-", color="#1f4e79")
ax[1].set_ylim(0, 200); ax[1].set_xlabel(r"$\ell_1$ budget $B$")
ax[1].set_ylabel(r"$\mathbb{E}[\tau]$")
ax[1].set_title(r"(b) budget is vacuous")
ax[1].axhline(116.7, ls=":", lw=0.8, color="0.4")

ax[2].loglog(SIG_D, SIG_T, "s-", color="#1f4e79", label=r"measured")
ref = np.array(SIG_D, float)
ax[2].loglog(ref, SIG_T[1] * (1.0 / ref) ** 2, "--", lw=0.9, color="0.45",
             label=r"$\propto 1/\Delta^2$")
ax[2].set_xlabel(r"signal $\Delta$"); ax[2].set_ylabel(r"$\mathbb{E}[\tau]$")
ax[2].set_title(r"(c) $\Gamma^\star\propto\Delta^2$")
ax[2].legend(frameon=False, fontsize=6.5)

fig.tight_layout(pad=0.4)
fig.savefig("fig_ablation.pdf"); fig.savefig("fig_ablation.png")
print("wrote fig_ablation.pdf / fig_ablation.png")

# ---- checks -------------------------------------------------------------
assert len(set(BUD_T)) == 1, "budget sweep must be exactly flat"
rt = [SIG_T[i] * SIG_G[i] for i in range(3)]
print("tau*Gamma* across the signal sweep:",
      [f"{v:.2f}" for v in rt], f"(spread {max(rt)/min(rt):.2f}x)")
assert max(rt) / min(rt) < 1.3, "tau*Gamma* should be roughly constant in Delta"
print("checks passed")
