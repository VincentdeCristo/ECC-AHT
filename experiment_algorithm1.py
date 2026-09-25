"""Algorithm 1 trajectories, parameter sweeps, and sensing/exploration ablations.

All candidates are enumerated. Cached projections accelerate the exact same
likelihood updates; the true set never enters sensing or the stopping rule.
"""
import argparse
import gzip
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import cvxpy
from revised_fixed_confidence import (
    ecc_pair, hypothesis_means, make_sigma, optimal_design, aggregate,
    verified_decision, update_exact_scores, run_trial,
)


def prepare(K, n, covariance, rho, signal, B, variant):
    sigma = make_sigma(K, covariance, rho)
    np.linalg.cholesky(sigma)
    hypotheses, means = hypothesis_means(K, n, signal)
    actions = list(B * np.eye(K))
    precision = np.linalg.inv(sigma)
    for i in range(K):
        for j in range(K):
            d = np.zeros(K)
            d[i] += signal
            d[j] -= signal
            if i == j:
                d[i] = signal  # unused diagonal entries keep cache rectangular
            # These covariance families act as a scalar on every contrast.
            # Use the analytic direction: inverse roundoff otherwise creates
            # spurious reference weights that break exact pseudo-score ties.
            c = d if variant == 'simple_diff' or covariance in {'identity', 'equicorr'} else precision @ d
            actions.append(B * c / np.linalg.norm(c, 1))
    actions = np.asarray(actions)
    std = np.sqrt(np.einsum('ik,kl,il->i', actions, sigma, actions))
    projected = actions @ means.T / std[:, None]
    per_stream = signal * actions / std[:, None]
    return sigma, hypotheses, means, actions, std, projected, per_stream


def trial_cached(prepared, true_index, n, delta, eta, cap, seed, variant='ecc', trace=False):
    sigma, hypotheses, means, actions, std, projected, per_stream = prepared
    M, K = means.shape
    rng = np.random.default_rng(seed)
    q, ell = np.zeros(M), np.zeros(K)
    boundary = math.log((M - 1) / delta)
    history = []
    for t in range(1, cap + 1):
        i, j = ecc_pair(ell, n)
        exploring = rng.random() < eta
        if variant == 'coordinate' or exploring:
            index = int(rng.integers(K))
        elif variant == 'random_pair':
            # Uniform signed two-coordinate projection, independent of scores.
            ii, jj = rng.choice(K, 2, replace=False)
            index = K + int(ii) * K + int(jj)
        else:
            index = K + i * K + j
        z = rng.normal(projected[index, true_index], 1.0)
        q -= 0.5 * (z - projected[index]) ** 2
        a = per_stream[index]
        ell += a * z - 0.5 * a * a
        decision = verified_decision(q, boundary)
        if trace:
            order = np.argsort(-q, kind='stable')
            history.append(dict(t=t, action=actions[index].tolist(),
                                y=float(z * std[index]), scores=ell.tolist(),
                                champion=i, challenger=j, exploration=bool(exploring),
                                likelihood_gap=float(q[order[0]]-q[order[1]])))
        if decision is not None:
            return dict(stopped=True, tau=t, correct=decision == true_index,
                        decision_index=decision, seed=seed, history=history)
    return dict(stopped=False, tau=cap, correct=False, decision_index=None,
                seed=seed, history=history)


def configurations():
    base = dict(K=20, n=2, covariance='toeplitz', rho=.5, signal=1., B=2.,
                eta=.1, variant='ecc', subset=[0, 1])
    cases = []
    def add(group, **kw):
        cases.append(dict(base, group=group, delta=.05, **kw))
    # Same likelihood protocol for every row; only the named factor changes.
    for covariance, rho in [('identity', 0.), ('equicorr', .9),
                            ('toeplitz', .5), ('toeplitz', .9)]:
        for variant in ['ecc', 'simple_diff', 'coordinate', 'random_pair', 'no_exploration']:
            add('ablation', covariance=covariance, rho=rho, variant=variant,
                eta=0. if variant == 'no_exploration' else .1)
    for B in [1., 2., 4.]:
        add('budget', B=B)
    for signal in [.5, 1., 2.]:
        add('signal', signal=signal)
    for n in [1, 2, 3, 4]:
        # 12 choose 4 = 495, allowing exact checks throughout this sweep.
        add('anomalies', K=12, n=n, subset=list(range(n)))
    for rho in [.2, .4, .6, .8]:
        for covariance in ['equicorr', 'toeplitz']:
            add('correlation', covariance=covariance, rho=rho)
    for delta in [.1, .03, .01, .003, .001]:
        case = dict(base, group='confidence', delta=delta)
        cases.append(case)
    for subset in [[0, 1], [0, 10]]:
        add('geometry', subset=subset)
    return cases


def selftest():
    prepared = prepare(4, 2, 'toeplitz', .5, 1., 2., 'ecc')
    sigma, hypotheses, means, actions, std, projected, _ = prepared
    for index in [0, 5, 7, 12]:
        q = np.zeros(len(means))
        y = .314
        update_exact_scores(q, actions[index], y, means, sigma)
        np.testing.assert_allclose(q, -.5 * (y/std[index]-projected[index])**2, atol=1e-12)
    for seed in range(10):
        cached = trial_cached(prepared, 2, 2, .05, .1, 500, seed, trace=True)
        reference = run_trial('ecc', sigma, means, hypotheses, 2, [], .05, 2., .1,
                              500, np.random.default_rng(seed))
        assert (cached['tau'], cached['decision_index']) == (reference['tau'], reference['decision_index'])
        q = np.zeros(len(means))
        for row in cached['history']:
            update_exact_scores(q, np.array(row['action']), row['y'], means, sigma)
            decision = verified_decision(q, math.log(5/.05))
            if row['t'] < cached['tau']:
                assert decision is None
        assert decision == cached['decision_index']
    # Uniform scaling must leave coupled stopping times unchanged in this model.
    for B in [1., 4.]:
        alternate = prepare(4, 2, 'toeplitz', .5, 1., B, 'ecc')
        row = trial_cached(alternate, 2, 2, .05, .1, 500, 17)
        ref = trial_cached(prepared, 2, 2, .05, .1, 500, 17)
        assert (row['tau'], row['decision_index']) == (ref['tau'], ref['decision_index'])
    for covariance, rho in [('identity', 0.), ('equicorr', .9)]:
        exact = prepare(20, 2, covariance, rho, 1., 2., 'ecc')
        plain = prepare(20, 2, covariance, rho, 1., 2., 'simple_diff')
        np.testing.assert_array_equal(exact[3], plain[3])
    print('PASS: cached likelihoods, independent score replay, scalar Algorithm 1 parity, budget invariance')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=300)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--groups', default='all')
    parser.add_argument('--out', default='results_algorithm1/experiments.json.gz')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.seeds < 2 or args.max_steps < 1:
        parser.error('require seeds >= 2 and max-steps >= 1')
    cases = configurations()
    if args.groups != 'all':
        cases = [c for c in cases if c['group'] in args.groups.split(',')]
    output = dict(protocol='Algorithm 1; exact enumeration; b=log((M-1)/delta)',
                  seed_policy='SHA256(config excluding B,variant,eta)+replicate; paired budget/ablation seeds',
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                scipy=scipy.__version__, cvxpy=cvxpy.__version__),
                  args=vars(args), cases=[])
    output['source_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for p in ['experiment_algorithm1.py', 'revised_fixed_confidence.py']}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rates = {}
    for case in cases:
        prep_variant = 'simple_diff' if case['variant'] == 'random_pair' else case['variant']
        prepared = prepare(**{k:case[k] for k in ['K','n','covariance','rho','signal','B']},
                           variant=prep_variant)
        sigma, hypotheses, means = prepared[:3]
        true_index = hypotheses.index(tuple(case['subset']))
        rate_key = (case['K'],case['n'],case['covariance'],case['rho'],case['signal'],true_index)
        if rate_key not in rates:
            design = optimal_design(sigma, means, true_index, case['B'])
            rates[rate_key] = {k: design[k] for k in ['gamma_star','achieved_rate','rate_error']}
        digest = hashlib.sha256(json.dumps({k:v for k,v in case.items() if k not in {'B','variant','eta'}},
                                          sort_keys=True).encode()).digest()
        seed_base = int.from_bytes(digest[:4], 'little')
        rows = []
        start = time.perf_counter()
        for replicate in range(args.seeds):
            row = trial_cached(prepared, true_index, case['n'], case['delta'], case['eta'],
                               args.max_steps, seed_base+replicate, variant=case['variant'])
            row.pop('history')
            rows.append(row)
        elapsed = time.perf_counter()-start
        summary = aggregate(rows, rates[rate_key]['gamma_star'], case['delta'])
        taus = np.array([r['tau'] for r in rows])
        summary.update(mean_tau_se=float(taus.std(ddof=1)/math.sqrt(args.seeds)),
                       simulation_seconds=elapsed, hypothesis_count=len(hypotheses))
        output['cases'].append(dict(config=case, summary=summary, trials=rows,
                                    design_diagnostic=rates[rate_key]))
        with gzip.open(out, 'wt', encoding='utf-8') as handle:
            json.dump(output, handle)
        print(f"{case['group']} {case['covariance']} rho={case['rho']} "
              f"{case['variant']} n={case['n']} d={case['delta']} B={case['B']} "
              f"signal={case['signal']} tau={summary['mean_tau_capped']:.2f} "
              f"wrong={summary['wrong_outputs']} timeout={summary['timeouts']} "
              f"seconds={elapsed:.1f}", flush=True)
    trace_case = prepare(15, 3, 'toeplitz', .6, 1., 2., 'ecc')
    trace = trial_cached(trace_case, trace_case[1].index((0,7,14)), 3, .05, .1,
                         args.max_steps, 30725, trace=True)
    output['trajectory'] = dict(config=dict(K=15,n=3,rho=.6,covariance='toeplitz',
                                           signal=1.,B=2.,eta=.1,delta=.05,subset=[0,7,14]), **trace)
    with gzip.open(out, 'wt', encoding='utf-8') as handle:
        json.dump(output, handle)
    print(f'wrote {out}: {len(output["cases"])} cases', flush=True)


if __name__ == '__main__':
    main()
