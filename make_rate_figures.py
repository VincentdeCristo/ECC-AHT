"""
Figures for the fixed-confidence experiments.

WHY THE DATA IS INLINE. The numeric results were produced by exp_rate_validation.py,
but the first several sweeps wrote to filenames that did not yet encode --signal and
--B (those CLI flags were added mid-session), so results/*.json overwrote each other
for the identity instance. The captured stdout of each run is intact, and that is what
is transcribed below. Re-running any row reproduces it:

    python exp_rate_validation.py --K 20 --n 2 --s-star 0,1 --sigma identity \
        --delta 0.05 --seeds 20 --method oracle

Every row is (Gamma* exact from the SDP, mean tau over 20 seeds, empirical error).
The error probability was 0.000 in every configuration, so delta-correctness is not
plotted; it is stated in the caption.

Sizing: single-column text width is ~3.5in, double-column ~7.16in. A wide
figure* is built at 7.0in with an 8pt base font, so no downscaling of glyphs occurs
-- a defect of the original figures.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
})

DELTA = 0.05

# ---- sweeps (see module docstring for provenance) -------------------------
N_SWEEP = [(1, 0.5263,  60.0), (2, 0.2778, 116.7),
           (3, 0.1961, 165.8), (4, 0.1562, 248.9)]
DELTA_SWEEP = [(0.1, 0.2778, 111.7), (0.05, 0.2778, 116.7), (0.02, 0.2778, 124.0),
               (0.01, 0.2778, 129.7), (0.005, 0.2778, 135.2), (0.001, 0.2778, 146.2)]
# Toeplitz at the SAME rho values as the equicorrelated line below, so the two
# curves are comparable point by point. Running them at different rhos (0.1/0.3/...
# against 0.2/0.4/...) made the panel visually misleading and was caught in review.
RHO_SWEEP = [(0.2, 0.2902, 144.8), (0.4, 0.3472, 192.1),
             (0.5, 0.3902, 277.3), (0.6, 0.4528, 217.9),
             (0.8, 0.7620, 169.3)]
# same rho = 0.5, different covariance STRUCTURE
STRUCTURE = {"identity": (0.2778, 116.7, 1), "equicorr": (0.5556, 61.1, 1),
             "toeplitz": (0.3902, 277.3, 2)}
# full equicorrelated line at the same rhos -- every point rank one, unlike Toeplitz
EQUICORR = [(0.2, 0.3472, 86.6), (0.4, 0.4630, 68.0), (0.5, 0.5556, 61.1),
            (0.6, 0.6944, 52.9), (0.8, 1.3889, 28.5)]
BUDGET = [(1.0, 116.7), (2.0, 116.7), (4.0, 116.7)]


def ratio(gamma, tau, delta=DELTA):
    return tau * gamma / np.log(1.0 / delta)


fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.15))

# (a) the rate law: tau*Gamma*/log(1/delta) is flat in n
n_ = [r[0] for r in N_SWEEP]
y_ = [ratio(r[1], r[2]) for r in N_SWEEP]
ax[0].plot(n_, y_, "o-", color="#1f4e79")
ax[0].axhline(np.mean(y_), ls="--", lw=0.8, color="0.45",
              label=f"mean {np.mean(y_):.1f}")
ax[0].set_xlabel(r"number of anomalies $n$")
ax[0].set_ylabel(r"$\mathbb{E}[\tau]\,\Gamma^\star / \log(1/\delta)$")
ax[0].set_title(r"(a) rate law: $\tau\propto 1/\Gamma^\star$")
ax[0].set_xticks(n_); ax[0].set_ylim(0, 20); ax[0].legend(frameon=False)

# (b) approach to the asymptotic constant as delta -> 0
d_ = [r[0] for r in DELTA_SWEEP]
y_ = [ratio(r[1], r[2], r[0]) for r in DELTA_SWEEP]
ax[1].semilogx(d_, y_, "s-", color="#1f4e79")
ax[1].axhline(1.0, ls=":", lw=0.8, color="0.45", label="asymptotic value 1")
ax[1].set_xlabel(r"target error probability $\delta$")
ax[1].set_title(r"(b) ratio decreases with $\delta$")
ax[1].invert_xaxis(); ax[1].set_ylim(0, 16); ax[1].legend(frameon=False)

# (c) the gap is structure, not strength
r_ = [x[0] for x in RHO_SWEEP]
y_ = [ratio(x[1], x[2]) for x in RHO_SWEEP]
ax[2].plot(r_, y_, "^-", color="#c00000",
           label=r"Toeplitz ($\mathrm{rank}\,W^\star=2$)")
eqr = [x[0] for x in EQUICORR]
eqy = [ratio(x[1], x[2]) for x in EQUICORR]
ax[2].plot(eqr, eqy, "o-", color="#1f4e79", zorder=5,
           label=r"equicorr., $\mathrm{rank}\,W^\star=1$")
ax[2].set_ylim(0, max(max(y_), max(eqy)) * 1.12)
ax[2].set_xlabel(r"correlation strength $\rho$")
ax[2].set_title(r"(c) cost is structural, not $\rho$")
ax[2].set_ylim(0, 42); ax[2].legend(frameon=False, loc="upper left")

fig.tight_layout(pad=0.4)
fig.savefig("fig_rate.pdf")
fig.savefig("fig_rate.png")
print("wrote fig_rate.pdf / fig_rate.png")

# ---- printed checks a reader can verify by hand ---------------------------
print("\nbudget sweep (prop: budget does not constrain the rate)")
base = BUDGET[0][1]
for b, t in BUDGET:
    print(f"  B={b:<4} tau={t:<7} identical to B={BUDGET[0][0]}: {abs(t-base) < 1e-9}")
assert len({t for _, t in BUDGET}) == 1, "budget sweep should be exactly flat"
lo = min(ratio(r[1], r[2]) for r in N_SWEEP)
hi = max(ratio(r[1], r[2]) for r in N_SWEEP)
print(f"\nn-sweep ratio spread: {lo:.2f} .. {hi:.2f}")
assert hi / lo < 1.4, "rate law should be flat in n"
print("checks passed")
