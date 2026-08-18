"""Lesson 3.12 - what a success rate actually claims.

Three functions are yours. The plumbing (the exact-test table, the printing) is
done. Run:  python trial_math.py

Expect the whole thing to take about ten seconds.
"""
import numpy as np
from scipy.stats import hypergeom, norm

Z95 = norm.ppf(0.975)          # 1.9600 - the 95% two-sided z


# --------------------------------------------------------------- intervals
def wilson_interval(k, n, z=Z95):
    """95% Wilson score interval for k successes out of n.

    TODO(you): with p = k/n,

        denominator = 1 + z^2/n
        centre      = (p + z^2/(2n)) / denominator
        half-width  = z/denominator * sqrt( p(1-p)/n + z^2/(4n^2) )

    Return (centre - half, centre + half), clipped to [0, 1].
    Write it with numpy so it also works on an ARRAY of k - lesson 3.13 needs
    that for the coverage check, and it costs you nothing here.
    """
    raise NotImplementedError


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
    """Trials PER POLICY to detect a p1-vs-p2 gap, at the given power.

    TODO(you): the standard two-proportion sample size. With
    p_bar = (p1+p2)/2, z_alpha = norm.ppf(1 - alpha/2), z_beta = norm.ppf(power):

        n = ( z_alpha*sqrt(2*p_bar*(1-p_bar))
              + z_beta*sqrt(p1*(1-p1) + p2*(1-p2)) )^2 / (p1-p2)^2

    Return it rounded UP to an integer. Before you run it, write down your guess
    for how many trials it takes to separate 90% from 80%. Then look.
    """
    raise NotImplementedError


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
    # TODO(you): walk n = step, 2*step, ... up to n_max. At each stop, look up
    # the Fisher p-value for the running counts a_cum[:, n-1] and b_cum[:, n-1]
    # (fisher_p_table(n) indexes with those arrays directly) and OR `declared`
    # with `p < alpha`. Once an experiment has declared a winner it stays
    # declared - that is exactly the behaviour of a person who stops when the
    # result looks good.
    raise NotImplementedError
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
    main()
