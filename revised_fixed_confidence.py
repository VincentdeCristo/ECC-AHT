"""Experiments for the revised fixed-confidence procedures.

This module intentionally enumerates all hypotheses.  It implements the
non-oracle two-stage construction used in the achievability theorem, the
verified ECC-AHT heuristic, and an oracle-design reference.  Every reported
decision uses exact subset likelihoods and the same all-alternative boundary.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from itertools import combinations
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.stats import beta


def make_sigma(K: int, kind: str, rho: float) -> np.ndarray:
    if kind == "identity":
        return np.eye(K)
    if kind == "equicorr":
        return (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    if kind == "toeplitz":
        idx = np.arange(K)
        return rho ** np.abs(idx[:, None] - idx[None, :])
    raise ValueError(f"unknown covariance: {kind}")


def hypothesis_means(K: int, n: int, signal: float) -> tuple[list[tuple[int, ...]], np.ndarray]:
    hypotheses = list(combinations(range(K), n))
    means = np.zeros((len(hypotheses), K))
    for row, subset in enumerate(hypotheses):
        means[row, list(subset)] = signal
    return hypotheses, means


def divergence(c: np.ndarray, d: np.ndarray, sigma: np.ndarray) -> float:
    return float(c @ d) ** 2 / (2.0 * float(c @ sigma @ c))


def optimal_design(
    sigma: np.ndarray,
    means: np.ndarray,
    hypothesis_index: int,
    budget: float,
    tolerance: float = 5e-4,
) -> dict:
    """Solve the rate SDP and return its finite eigen-support realization."""
    K = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)
    linv = np.linalg.inv(chol)
    keep = np.arange(len(means)) != hypothesis_index
    differences = means[hypothesis_index] - means[keep]
    whitened = differences @ linv.T

    W = cp.Variable((K, K), symmetric=True)
    g = cp.Variable()
    constraints = [W >> 0, cp.trace(W) == 1]
    constraints.extend(cp.quad_form(v, W) >= g for v in whitened)
    problem = cp.Problem(cp.Maximize(g), constraints)
    problem.solve(solver=cp.SCS, eps=1e-8, max_iters=200_000)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or W.value is None:
        raise RuntimeError(f"SDP failed: {problem.status}")

    W_value = (np.asarray(W.value) + np.asarray(W.value).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(W_value)
    positive = eigenvalues > max(1e-9, 1e-7 * float(np.max(eigenvalues)))
    probabilities = np.maximum(eigenvalues[positive], 0.0)
    probabilities /= probabilities.sum()
    white_directions = eigenvectors[:, positive].T
    actions = []
    for w in white_directions:
        c = np.linalg.solve(chol.T, w)
        c *= budget / np.linalg.norm(c, 1)
        actions.append(c)
    actions = np.asarray(actions)

    reconstructed = sum(p * np.outer(w, w) for p, w in zip(probabilities, white_directions))
    achieved = min(
        sum(p * divergence(c, d, sigma) for p, c in zip(probabilities, actions))
        for d in differences
    )
    gamma_star = float(g.value) / 2.0
    reconstruction_error = float(np.linalg.norm(reconstructed - W_value, ord="fro"))
    rate_error = abs(achieved - gamma_star)
    allowed = max(tolerance, tolerance * abs(gamma_star))
    if reconstruction_error > 5 * tolerance or rate_error > allowed:
        raise RuntimeError(
            f"invalid SDP realization: reconstruction={reconstruction_error:.3g}, "
            f"rate error={rate_error:.3g}"
        )
    return {
        "gamma_star": gamma_star,
        "probabilities": probabilities,
        "actions": actions,
        "rank": int(len(probabilities)),
        "reconstruction_error": reconstruction_error,
        "achieved_rate": achieved,
        "rate_error": rate_error,
    }


def update_exact_scores(
    scores: np.ndarray,
    c: np.ndarray,
    y: float,
    means: np.ndarray,
    sigma: np.ndarray,
) -> None:
    variance = max(float(c @ sigma @ c), 1e-14)
    residuals = y - means @ c
    scores -= residuals * residuals / (2.0 * variance)


def verified_decision(scores: np.ndarray, boundary: float) -> int | None:
    order = np.argsort(-scores, kind="stable")
    best = int(order[0])
    return best if scores[best] - scores[order[1]] >= boundary else None


def draw_design_action(design: dict, rng: np.random.Generator) -> np.ndarray:
    index = int(rng.choice(len(design["probabilities"]), p=design["probabilities"]))
    return design["actions"][index]


def observe(
    c: np.ndarray, true_mean: np.ndarray, sigma: np.ndarray, rng: np.random.Generator
) -> float:
    variance = max(float(c @ sigma @ c), 1e-14)
    return float(rng.normal(float(c @ true_mean), math.sqrt(variance)))


def run_trial(
    method: str,
    sigma: np.ndarray,
    means: np.ndarray,
    hypotheses: list[tuple[int, ...]],
    true_index: int,
    designs: list[dict],
    delta: float,
    budget: float,
    eta: float,
    max_steps: int,
    rng: np.random.Generator,
) -> dict:
    M, K = means.shape
    n = len(hypotheses[0])
    boundary = math.log((M - 1) / delta)
    exact_scores = np.zeros(M)
    pseudo_scores = np.zeros(K)
    true_mean = means[true_index]
    signal = float(np.max(means))
    t = 0

    def take(c: np.ndarray, update_pseudo: bool = False) -> int | None:
        nonlocal t
        t += 1
        y = observe(c, true_mean, sigma, rng)
        update_exact_scores(exact_scores, c, y, means, sigma)
        if update_pseudo:
            variance = max(float(c @ sigma @ c), 1e-14)
            a = signal * c
            pseudo_scores[:] += (a * y - 0.5 * a * a) / variance
        return verified_decision(exact_scores, boundary)

    if method == "two_stage":
        h = max(1.0, boundary)
        q = int(math.ceil(math.sqrt(h)))
        epsilon = min(0.5, h ** (-0.25))
        for k in range(K):
            c = np.zeros(K)
            c[k] = budget
            for _ in range(q):
                if t >= max_steps:
                    return {
                        "stopped": False,
                        "tau": max_steps,
                        "decision_index": None,
                        "correct": False,
                        "q": q,
                        "epsilon": epsilon,
                    }
                take(c)
        pilot_choice = int(np.argmax(exact_scores))
        while t < max_steps:
            if rng.random() < epsilon:
                c = np.zeros(K)
                c[int(rng.integers(K))] = budget
            else:
                c = draw_design_action(designs[pilot_choice], rng)
            decision = take(c)
            if decision is not None:
                return {
                    "stopped": True,
                    "tau": t,
                    "decision_index": decision,
                    "correct": bool(decision == true_index),
                    "pilot_index": pilot_choice,
                    "pilot_correct": bool(pilot_choice == true_index),
                    "q": q,
                    "epsilon": epsilon,
                }

    elif method == "oracle_design":
        while t < max_steps:
            decision = take(draw_design_action(designs[true_index], rng))
            if decision is not None:
                return {
                    "stopped": True,
                    "tau": t,
                    "decision_index": decision,
                    "correct": bool(decision == true_index),
                }

    elif method == "verified_ecc":
        while t < max_steps:
            if rng.random() < eta:
                c = np.zeros(K)
                c[int(rng.integers(K))] = budget
            else:
                ranking = np.argsort(-pseudo_scores, kind="stable")
                selected = ranking[:n]
                rejected = ranking[n:]
                i = int(selected[np.argmin(pseudo_scores[selected])])
                j = int(rejected[np.argmax(pseudo_scores[rejected])])
                d = np.zeros(K)
                d[i], d[j] = signal, -signal
                c = np.linalg.solve(sigma, d)
                c *= budget / np.linalg.norm(c, 1)
            decision = take(c, update_pseudo=True)
            if decision is not None:
                return {
                    "stopped": True,
                    "tau": t,
                    "decision_index": decision,
                    "correct": bool(decision == true_index),
                }
    else:
        raise ValueError(f"unknown method: {method}")

    return {
        "stopped": False,
        "tau": max_steps,
        "decision_index": None,
        "correct": False,
    }


def clopper_pearson_upper(errors: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return float("nan")
    if errors >= trials:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, trials - errors))


def aggregate(trials: list[dict], gamma_star: float, delta: float) -> dict:
    n_trials = len(trials)
    stopped = [r for r in trials if r["stopped"]]
    wrong = sum(bool(r["stopped"] and not r["correct"]) for r in trials)
    capped = np.asarray([r["tau"] for r in trials], dtype=float)
    stopped_tau = np.asarray([r["tau"] for r in stopped], dtype=float)
    mean_capped = float(capped.mean())
    mean_stopped = float(stopped_tau.mean()) if len(stopped_tau) else None
    return {
        "trials": n_trials,
        "stopped": len(stopped),
        "timeouts": n_trials - len(stopped),
        "wrong_outputs": wrong,
        "error_rate_all_trials": wrong / n_trials,
        "error_rate_stopped": wrong / len(stopped) if stopped else None,
        "error_95pct_upper_all_trials": clopper_pearson_upper(wrong, n_trials),
        "mean_tau_capped": mean_capped,
        "median_tau_capped": float(np.median(capped)),
        "mean_tau_stopped": mean_stopped,
        "gamma_star": gamma_star,
        "normalized_mean_tau_capped": mean_capped * gamma_star / math.log(1.0 / delta),
        "normalized_mean_tau_stopped": (
            mean_stopped * gamma_star / math.log(1.0 / delta)
            if mean_stopped is not None
            else None
        ),
        "timeout_note": "Capped summaries include max_steps for every timeout.",
    }


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--methods", default="two_stage,verified_ecc,oracle_design")
    parser.add_argument("--deltas", default="0.1,0.05,0.01")
    parser.add_argument("--covariances", default="identity,toeplitz")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--signal", type=float, default=1.0)
    parser.add_argument("--B", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30725)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--s-star", default="", help="comma-separated true subset")
    parser.add_argument("--out", default="results_revised/revised_fixed_confidence.json.gz")
    args = parser.parse_args()

    if not (1 <= args.n < args.K):
        parser.error("require 1 <= n < K")
    if not (0 < args.eta < 1 and 0 < args.B and 0 < args.signal
            and args.seeds > 0 and args.max_steps > 0):
        parser.error("eta, B, signal, seeds, and max-steps must be positive (eta < 1)")

    methods = parse_csv(args.methods)
    deltas = parse_csv(args.deltas, float)
    covariance_names = parse_csv(args.covariances)
    allowed_methods = {"two_stage", "verified_ecc", "oracle_design"}
    if not set(methods) <= allowed_methods or any(not 0 < d < 0.5 for d in deltas):
        parser.error("invalid method or delta")

    hypotheses, means = hypothesis_means(args.K, args.n, args.signal)
    if args.s_star:
        true_subset = tuple(sorted(parse_csv(args.s_star, int)))
    else:
        true_subset = tuple(range(args.n))
    if true_subset not in hypotheses:
        parser.error("s-star must name exactly n distinct indices in [0,K)")
    true_index = hypotheses.index(true_subset)

    output = {
        "schema_version": 1,
        "created_unix": time.time(),
        "protocol": "exact all-hypothesis likelihood; constant boundary log((M-1)/delta)",
        "config": vars(args),
        "true_subset": list(true_subset),
        "hypothesis_count": len(hypotheses),
        "design_diagnostics": {},
        "aggregates": [],
        "trials": [],
    }

    configuration_index = 0
    for covariance_name in covariance_names:
        sigma = make_sigma(args.K, covariance_name, args.rho)
        designs = [optimal_design(sigma, means, i, args.B) for i in range(len(hypotheses))]
        output["design_diagnostics"][covariance_name] = [
            {
                "hypothesis": list(hypotheses[i]),
                "gamma_star": d["gamma_star"],
                "rank": d["rank"],
                "reconstruction_error": d["reconstruction_error"],
                "achieved_rate": d["achieved_rate"],
                "rate_error": d["rate_error"],
            }
            for i, d in enumerate(designs)
        ]
        gamma_true = designs[true_index]["gamma_star"]

        for delta in deltas:
            for method in methods:
                rows = []
                for trial_index in range(args.seeds):
                    rng = np.random.default_rng(
                        np.random.SeedSequence([args.seed, configuration_index, trial_index])
                    )
                    row = run_trial(
                        method, sigma, means, hypotheses, true_index, designs,
                        delta, args.B, args.eta, args.max_steps, rng,
                    )
                    row.update(
                        covariance=covariance_name,
                        rho=args.rho,
                        delta=delta,
                        method=method,
                        trial=trial_index,
                    )
                    rows.append(row)
                summary = aggregate(rows, gamma_true, delta)
                summary.update(covariance=covariance_name, rho=args.rho, delta=delta, method=method)
                output["aggregates"].append(summary)
                output["trials"].extend(rows)
                print(
                    f"{covariance_name:10s} delta={delta:g} {method:14s} "
                    f"tau_cap={summary['mean_tau_capped']:.2f} "
                    f"err={summary['error_rate_all_trials']:.4f} "
                    f"upper95={summary['error_95pct_upper_all_trials']:.4f} "
                    f"timeouts={summary['timeouts']} ratio={summary['normalized_mean_tau_capped']:.3f}"
                )
                configuration_index += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(output, indent=2)
    if out.suffix == ".gz":
        with gzip.open(out, "wt", encoding="utf-8") as handle:
            handle.write(serialized)
    else:
        out.write_text(serialized, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
