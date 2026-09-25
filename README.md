# Active Hypothesis Testing for Correlated Combinatorial Anomaly Detection

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2601.17430-red?logo=arxiv)](https://arxiv.org/abs/2601.17430)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Paper](https://img.shields.io/badge/Hugging%20Face-paper-yellow?logo=huggingface)](https://huggingface.co/papers/2601.17430)

**[Paper](https://arxiv.org/abs/2601.17430)** | **[Hugging Face](https://huggingface.co/papers/2601.17430)**

</div>

This repository contains the official implementation of ECC-AHT, a sequential
active hypothesis testing algorithm for correlated combinatorial anomaly detection.

## 📰 News

- **[1/26/2026]** Preprint available on [arXiv](https://arxiv.org/abs/2601.17430)
- **[1/24/2026]** Code and experiments released

## ⭐ Overview

ECC-AHT addresses the problem of identifying a small set of anomalous streams under
correlated Gaussian noise. The algorithm combines:

- correlation-aware measurement design,
- Champion–Challenger hypothesis comparison,
- scalable pseudo-likelihood inference.

The revised theory separates two procedures. An enumerative, non-oracle two-stage
construction attains the controlled-sensing information rate for fixed finite
hypothesis families. Verified ECC-AHT uses persistent coordinate exploration and
exact all-hypothesis likelihood verification; it is delta-correct with finite mean
stopping time, but no rate-optimality theorem is claimed for this heuristic.

<img width="1135" height="586" alt="image" src="https://github.com/user-attachments/assets/5277dadf-3e11-4660-836e-900366afb464" />

## 📄 License

This project is licensed under the [MIT License](https://github.com/VincentdeCristo/ECC-AHT/blob/main/LICENSE).

## 🚀 Quick Start

### 🧪 Configure environment

```bash
mamba create -n eccaht python=3.12.11
mamba activate eccaht
mamba install --file requirements.txt
```

### Revised procedures

`revised_fixed_confidence.py` implements the revised theorem and algorithm under
one exact decision protocol. It enumerates all size-`n` hypotheses, so use small
`K` unless the resulting combinatorial cost is acceptable.

```bash
# Quick deterministic smoke run
python revised_fixed_confidence.py --K 4 --n 2 --s-star 0,3 \
  --covariances identity --deltas 0.2 --seeds 2 --max-steps 1000 \
  --out results_revised/smoke.json.gz

# Fixed-confidence coverage table used for the revised manuscript
python revised_fixed_confidence.py --K 4 --n 2 --s-star 0,3 \
  --covariances identity,toeplitz --rho 0.5 \
  --deltas 0.05,0.01 --seeds 3000 --max-steps 5000 \
  --out results_revised/coverage.json.gz

# Asymptotic-rate figure used for the revised manuscript
python revised_fixed_confidence.py --K 4 --n 2 --s-star 0,3 \
  --covariances identity,toeplitz --rho 0.5 \
  --deltas 1e-3,1e-6,1e-12,1e-24,1e-48 --seeds 300 --max-steps 5000 \
  --out results_revised/rate_tail.json.gz

python make_revised_fixed_confidence_figure.py results_revised/rate_tail.json.gz

python -m unittest -v test_revised_fixed_confidence.py
```

The three methods are `two_stage` (the non-oracle attaining construction),
`verified_ecc` (the practical heuristic), and `oracle_design` (a diagnostic
reference that knows the true set only for sensing). Every output decision is
the exact maximum-likelihood subset and must beat every alternative by
`log((M-1)/delta)`. JSON output includes raw trials, timeout counts, capped and
stopped-only sample summaries, the exact SDP rate, normalized sample size, and
a one-sided 95% Clopper--Pearson error bound.

### 📐 Legacy fixed-confidence re-analysis scripts

The scripts below were used to audit and re-express experiments from the earlier
implementation under fixed-confidence metrics. They retain legacy sensing or
inference components and do not implement the revised theorem or Algorithm 1.
Use `revised_fixed_confidence.py` above for results claimed for the revised
procedures. The older scripts remain available for traceability.

- **Figure 2** — rate law and its cost (3 panels)

  ```bash
  python exp_rate_validation.py --K 20 --n 2 --sigma toeplitz --delta 0.05
  python make_rate_figures.py
  ```

- **Figure 3** — design-rule comparison and two checks of the theory

  ```bash
  python ablation_fc.py
  python make_ablation_figure.py
  ```

  `ablation_fc.py` drives the baseline classes in `algorithms.py` (`CombGapE`,
  `TTTS`, `RSP`, ...) with the fixed-confidence stopping rule, so `algorithms.py`
  must be importable. One caveat: `mu_signal` must be a **scalar**, because
  `algorithms.py` assigns `delta_t[i_star] = self.delta_signal` and then
  multiplies elementwise.

- **Lower bound** `E[tau] >= d(delta, 1-delta) / Gamma*` (Theorem 6)

  ```bash
  python lower_bound_check.py
  ```

- **WaDi** legacy real-data diagnostic (not used as validation in the revised paper)

  ```bash
  python wadi_fc.py
  ```

- Supporting checks: `gamma_single_vs_mixture.py` (whether the single-design
  rate coincides with the mixture rate), `exact_certificate_identity.py` (exact
  rational certificate `Gamma_sd = Gamma* = 5/18` on the identity instance),
  `threshold_constant.py`, `measure_achieved_rate.py`, `diagnose_convergence.py`.

Raw per-run JSON from these sweeps is under `results_tit/`.

### 📊 Reproducing the arXiv preprint (v1) results

The figure numbers below refer to the original arXiv preprint, which differs
from the fixed-confidence protocol above. All experiments from that version can
be reproduced using scripts

- Scalability experiment (Figure 5) & Ablation study (Figure 2)

  ```bash
  python run_simulation.py
  ```

- SOTA comparison (Figure 3, 10 -- 16)

  ```bash
  python run_tree.py
  python run_sota.py
  ```

  *Note:* To Get **Figure 19 (a), 20(a), 21(a), 22 -- 26**, you should add a line

  ```python
  Sigma += 0.01 * np.eye(K)
  ```

  after line 222.
  To Get **Figure 19(b), 20(b), 21(b)**, you should change the `rho` from `0.8` to `0.5` in line 197 and comment out all other items in line 199 of correlation_modes except for `Equicorrelation`, `Kronecker`, and `RBF`.
- Robustness analysis (Figure 6 -- 9)

  ```bash
  python run_robustness.py
  ```

- Real-World evaluation (Figure 4)
  
  First of all, apply the dataset on its [official website](https://itrust.sutd.edu.sg/itrust-labs_datasets/).
  Then:

  ```bash
  python preprocess_wadi.py
  python run_wadi_a.py
  python run_wadi_b.py
  ```

- Interpretative analysis (Figure 1)

  ```bash
  python visualize_inside_ecc_aht.py
  ```

- Limitation analysis (Table 1, Figure 17, 18)

  ```bash
  python run_experiments_spectral_rank.py
  ```

## 📧 Contact

**Authors:**
- Zichuan Yang ([2153747@tongji.edu.cn](mailto:2153747@tongji.edu.cn))
- Yiming Xing ([yimingx4@tongji.edu.cn](mailto:yimingx4@tongji.edu.cn))

**Questions?** Open an [issue](https://github.com/VincentdeCristo/ECC-AHT/issues) or email us!

## 📖 Citation

If you use this code in your research, please cite:
```bibtex
@misc{yang2026activehypothesistestingcorrelated,
      title={Active Hypothesis Testing for Correlated Combinatorial Anomaly Detection}, 
      author={Zichuan Yang and Yiming Xing},
      year={2026},
      eprint={2601.17430},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.17430}, 
}
```
