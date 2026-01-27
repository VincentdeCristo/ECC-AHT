# Active Hypothesis Testing for Correlated Combinatorial Anomaly Detection

This repository contains the official implementation of ECC-AHT, a sequential
active hypothesis testing algorithm for correlated combinatorial anomaly detection.

[![Paper](https://img.shields.io/badge/arXiv-26.1-red?logo=arxiv)](https://arxiv.org/abs/2601.17430)

## 📰 News

- **[1/24/2026]** Code and experiments released

## ⭐ Overview

ECC-AHT addresses the problem of identifying a small set of anomalous streams under
correlated Gaussian noise. The algorithm combines:

- correlation-aware measurement design,
- Champion–Challenger hypothesis comparison,
- scalable pseudo-likelihood inference.

This allows ECC-AHT to achieve information-theoretically optimal rates while remaining
computationally efficient for large-scale systems.

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

### 📊 Reproducing Paper Results

All experiments from the paper can be reproduced using scripts

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
