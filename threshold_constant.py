"""Verify the asymptotic constant of the fixed-tau Gaussian-mixture threshold.

Canonical reduction (paper's own notation):
  rho_t = c_t^T u_{S'}/sigma_t,  V_t = sum_{s<=t} rho_s^2,  Z_t = sum_{s<=t} rho_s z_s,
  z_t iid N(0,1) under H_{S*},  E_{S'}[z_t] = rho_t.
  Threshold  c(V,d) = (1/tau) sqrt(2(1+tau^2 V)[log(1/d) + (1/2) log(1+tau^2 V)]).
  Rule: stop at first t with Z_t >= c(V_t, delta')   [one-sided; Ville is on |Z_t|].

Unit case used below: rho_t = 1  =>  V_t = t,  Z_t = drift*t + N(0,t), drift=1 under H_{S'}
and drift=0 under H_{S*}.  Per-step divergence D = rho^2/2 = 1/2.
"""
import numpy as np

rng = np.random.default_rng(12345)


def c_bnd(V, Lam, tau):
    s = 1.0 + tau * tau * V
    return np.sqrt(2.0 * s * (Lam + 0.5 * np.log(s))) / tau


def first_hit(Z, b, T):
    hit = Z >= b
    any_hit = hit.any(axis=1)
    idx = np.where(any_hit, hit.argmax(axis=1) + 1, T + 1)
    return idx


def run(drift, Lam, tau, R, T, mode="mix"):
    z = rng.normal(drift, 1.0, size=(R, T))
    Z = np.cumsum(z, axis=1)
    t = np.arange(1, T + 1, dtype=float)[None, :]
    if mode == "mix":
        b = c_bnd(t, Lam, tau)
    elif mode == "sprt":          # L_t = Z_t - V_t/2 >= Lam   <=>  Z_t >= t/2 + Lam
        b = t / 2.0 + Lam
    elif mode == "glr":           # Z_t^2/(2 V_t) >= Lam  (profile / GLR statistic)
        b = np.sqrt(2.0 * Lam * t)
    return first_hit(Z, b, T)


print("=" * 78)
print("VALIDITY under H_{S*} (drift=0): P(any crossing) must be <= delta' = e^{-Lam}")
print("=" * 78)
for Lam in [4.0, 6.0, 8.0]:
    idx = run(0.0, Lam, 1.0, 400000, 3000)
    print(f"  Lam={Lam:5.1f}  delta'=e^-Lam={np.exp(-Lam):.3e}  "
          f"measured P(cross)={(idx <= 3000).mean():.3e}")

print()
print("=" * 78)
print("POWER/TIME under H_{S'}: E[tau] for the mixture rule (tau=1) vs SPRT vs GLR")
print("prediction (mixture): V* = 2*Lam + log(2*tau^2*Lam),  V_t = t")
print("=" * 78)
print(f"{'Lam':>6} {'2Lam':>8} {'pred V*':>9} {'E[tau] mix':>11} {'E[tau] SPRT':>12} "
      f"{'E[tau] GLR':>11} {'ratio mix/Lam':>14}")
R = 40000
for Lam in [5.0, 10.0, 20.0, 40.0, 80.0]:
    T = int(4 * Lam + 200)
    m = run(1.0, Lam, 1.0, R, T, "mix")
    s = run(1.0, Lam, 1.0, R, T, "sprt")
    g = run(1.0, Lam, 1.0, R, T, "glr")
    pred = 2 * Lam + np.log(2 * Lam)
    print(f"{Lam:6.1f} {2*Lam:8.1f} {pred:9.2f} {m.mean():11.2f} {s.mean():12.2f} "
          f"{g.mean():11.2f} {m.mean()/Lam:14.3f}")

print()
print("=" * 78)
print("tau-SWEEP at Lam=20 (drift=1): only O(log tau) variation is possible")
print("=" * 78)
for tau in [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 20.0]:
    T = 300
    m = run(1.0, 20.0, tau, R // 2, T, "mix")
    pred = 2 * 20 + np.log(2 * tau * tau * 20)
    print(f"  tau={tau:6.2f}   E[tau]={m.mean():8.2f}   pred(2Lam+log(2tau^2Lam))={pred:8.2f}")

print()
print("=" * 78)
print("UNION-BOUND EXPLANATION of a measured ratio ~2.2 (K=20,n=2 => |A|=C(20,2)-1=189)")
print("  benchmark is log(|A|/delta), not log(1/delta)")
print("=" * 78)
with np.errstate(divide="ignore"):
    for delta in [0.5, 0.2, 0.1, 0.05, 0.01, 5e-3, 1e-3, 1e-4, 1e-6]:
        L = np.log(1 / delta)
        Lam = L + np.log(189.0)
        tot = 2 * Lam + np.log(2 * Lam)
        print(f"  delta={delta:8.1e}  log(1/d)={L:5.2f}  log(|A|/d)={Lam:6.2f}  "
              f"V*={tot:6.2f}  V*/log(1/d)={tot/L:5.2f}")
