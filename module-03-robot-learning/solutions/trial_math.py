"""Solution - Lesson 3.12: what a success rate actually claims.

Self-contained. Run:  python trial_math.py

Three blocks, in the order the lesson argues them:
  1. one success rate, and how wide it really is
  2. the "90% beats 80%" claim, and what it would cost to earn it
  3. what peeking at the running score does to your false-win rate
"""
import numpy as np
from scipy.stats import hypergeom, norm

Z95 = norm.ppf(0.975)          # 1.9600 - the 95% two-sided z


# --------------------------------------------------------------- intervals
def wilson_interval(k, n, z=Z95):
    """95% Wilson score interval for k successes out of n. Vectorised over k."""
    k = np.asarray(k, dtype=float)
    p_hat = k / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def wald_interval(k, n, z=Z95):
    """The textbook interval, kept only so lesson 3.13 can show it failing."""
    k = np.asarray(k, dtype=float)
    p_hat = k / n
    half = z * np.sqrt(p_hat * (1 - p_hat) / n)
    return np.clip(p_hat - half, 0.0, 1.0), np.clip(p_hat + half, 0.0, 1.0)


# ------------------------------------------------------------- comparisons
def fisher_p_table(n):
    """p[a, b] = two-sided Fisher exact p-value for [[a, n-a], [b, n-b]].

    Built once per n and then indexed, because the per-call scipy version is
    ~140x slower and the simulation below needs millions of look-ups.
    Conditioning on the column total s = a + b makes a hypergeometric.
    """
    out = np.ones((n + 1, n + 1))
    for s in range(2 * n + 1):
        support = np.arange(max(0, s - n), min(n, s) + 1)
        pmf = hypergeom.pmf(support, 2 * n, s, n)
        # Two-sided: total probability of every table no more likely than this one.
        p = np.array([pmf[pmf <= pmf[i] * (1 + 1e-7)].sum() for i in range(len(support))])
        for i, a in enumerate(support):
            out[a, s - a] = min(p[i], 1.0)
    return out


def trials_needed(p1, p2, power=0.80, alpha=0.05):
    """Trials PER POLICY to detect a p1-vs-p2 gap, at the given power."""
    p_bar = (p1 + p2) / 2
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    numer = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
             + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(numer / (p1 - p2) ** 2))


# ------------------------------------------------------------ the peek test
def false_win_rate(step, n_max=40, p_true=0.70, sims=100_000, seed=12, alpha=0.05):
    """Two IDENTICAL policies. How often do we ever declare a winner?

    `step` is how often we look at the running score. step == n_max is the
    honest protocol: fix n up front, test once, at the end.
    """
    rng = np.random.default_rng(seed)
    a_cum = np.cumsum(rng.random((sims, n_max)) < p_true, axis=1)
    b_cum = np.cumsum(rng.random((sims, n_max)) < p_true, axis=1)

    declared = np.zeros(sims, dtype=bool)
    for n in range(step, n_max + 1, step):
        table = fisher_p_table(n)
        declared |= table[a_cum[:, n - 1], b_cum[:, n - 1]] < alpha
    return declared.mean()


# ------------------------------------------------------------------- report
def main():
    print("1. ONE SUCCESS RATE, HONESTLY REPORTED   (observed 80% every time)")
    print(f"   {'n':>5}  {'k':>4}   95% Wilson interval      width")
    for n in (10, 20, 50, 100, 200, 500):
        k = round(0.8 * n)
        lo, hi = wilson_interval(k, n)
        print(f"   {n:5d}  {k:4d}   [{lo:.3f}, {hi:.3f}]   {hi - lo:>10.3f}")

    print("\n2. \"OUR POLICY GETS 90%, THE BASELINE GETS 80%\"")
    for n in (10, 20, 50, 100, 200):
        ka, kb = round(0.9 * n), round(0.8 * n)
        la, ha = wilson_interval(ka, n)
        lb, hb = wilson_interval(kb, n)
        p = fisher_p_table(n)[ka, kb]
        verdict = "overlap" if (la < hb and lb < ha) else "SEPARATED"
        print(f"   n={n:4d}  A [{la:.2f},{ha:.2f}]  B [{lb:.2f},{hb:.2f}]  "
              f"{verdict:9s}  Fisher p = {p:.3f}")
    print("   trials per policy needed to earn that claim at 80% power:")
    for gap in (0.05, 0.10, 0.15, 0.20):
        print(f"     0.90 vs {0.90 - gap:.2f}   {trials_needed(0.90, 0.90 - gap):>5d}")

    print("\n3. PEEKING AT THE RUNNING SCORE  (two identical policies, both 70%)")
    print("   40 trials each, 100k simulated experiments, alpha = 0.05")
    for step, label in ((40, "fix n=40, test once at the end"),
                        (10, "peek every 10 trials"),
                        (5, "peek every 5 trials"),
                        (1, "peek after every single trial")):
        print(f"   {label:34s} declares a winner {false_win_rate(step):6.1%} of the time")


if __name__ == "__main__":
    # Behavioural checks: these are the claims the lesson makes in prose.
    lo, hi = wilson_interval(8, 10)
    assert (round(lo, 2), round(hi, 2)) == (0.49, 0.94), (lo, hi)
    assert wilson_interval(10, 10)[1] > 0.999 and wilson_interval(10, 10)[0] > 0.7
    # Wald degenerates to a zero-width interval at k == n; Wilson does not.
    assert wald_interval(10, 10) == (1.0, 1.0)
    assert trials_needed(0.90, 0.80) == 199
    once, peeked = false_win_rate(40, sims=20_000), false_win_rate(1, sims=20_000)
    assert peeked > 2 * once, (once, peeked)
    print("solution trial_math: checks pass.\n")
    main()
