"""
p* and revenue-rate sensitivity to the abandonment rate gamma
===============================================================

Gamma-analogue of the existing "p* vs q" experiment. Instead of
re-implementing the simulation / fixed-point / golden-section-search
machinery, this script imports it directly from goldenmethod.py:

    simulate()
    fixed_point_solve()
    evaluate_price()
    golden_section_search()
    repeated_golden_section_search()

IMPORTANT: place this file in the same folder as goldenmethod.py (or make
sure goldenmethod.py is importable, e.g. on PYTHONPATH), since everything
below is built on top of it.

Only two things are added here:
  - sweep_gamma(): loops repeated_golden_section_search() over a grid of
    gamma values (lambda1, lambda2, q held fixed), collecting p* and the
    maximized revenue rate at each gamma.
  - plot_gamma_sweep(): plots p* vs gamma and revenue_rate* vs gamma
    side by side, with 95% CI error bars.

Economic intuition to check against the plot (see model_introduction.docx /
Buke & Chen FDAPMS Prop. 10): larger gamma -> users abandon faster -> for
any fixed p, the realized match probability pi_m(p) falls -> employers'
expected utility u*pi_m falls -> the platform is expected to lower price to
keep employers joining, i.e. p* should decrease in gamma, and the maximized
revenue rate should decrease too (possibly with a "cliff" like the one in
revenue_rate_vs_price1.png, just triggered at a lower price as gamma grows).
"""

import statistics
import matplotlib.pyplot as plt

# Reuse the existing simulation / search machinery as-is — nothing about
# the model or the search algorithm is reimplemented here.
from goldenmethod import golden_section_search


def _mean_std_ci95(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * std / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return mean, std, ci95


# ---------------------------------------------------------------------
# Sweep gamma, collect (p*, revenue_rate*) at each gamma
# ---------------------------------------------------------------------
def sweep_gamma(gamma_list, lambda1, lambda2, q, termination_time,
                 p_low, p_high, n_repeats=8, tol=1.0,
                 n_reps=20, eval_reps=40, use_crn=True, crn_reps=40,
                 verbose=True):
    """
    For each gamma in gamma_list, call golden_section_search() n_repeats
    times (independent CRN draws each time — same pattern as
    repeated_golden_section_search in goldenmethod.py), collecting both
    p* and revenue_rate* per repeat, then summarize each with mean/95% CI.
    Mirrors a "p* vs q" sweep, with gamma as the swept parameter and q
    held fixed.
    """
    results = []
    for gamma in gamma_list:
        if verbose:
            print(f"\n--- gamma = {gamma:.3f} "
                  f"({n_repeats} independent golden-section searches) ---")

        p_list = []
        rr_list = []
        pim_list = []
        revenue_list = []
        for r in range(n_repeats):
            p_star, pi_m_star, revenue_star, revenue_rate_star, _ = golden_section_search(
                lambda1, lambda2, q, gamma, termination_time,
                p_low=p_low, p_high=p_high, tol=tol,
                n_reps=n_reps, eval_reps=eval_reps,
                use_crn=use_crn, crn_reps=crn_reps,
                crn_seed=None, verbose=False)
            p_list.append(p_star)
            rr_list.append(revenue_rate_star)
            pim_list.append(pi_m_star)
            revenue_list.append(revenue_star)

            if verbose:
                print(f"  repeat {r + 1:2d}/{n_repeats}: "
                      f"p* = {p_star:7.3f}   "
                      f"pi_m* = {pi_m_star:.4f}   "
                      f"revenue* = {revenue_star:8.2f}   "
                      f"revenue_rate* = {revenue_rate_star:.4f}")

        p_mean, p_std, p_ci95 = _mean_std_ci95(p_list)
        rr_mean, rr_std, rr_ci95 = _mean_std_ci95(rr_list)
        pim_mean, pim_std, pim_ci95 = _mean_std_ci95(pim_list)
        rev_mean, rev_std, rev_ci95 = _mean_std_ci95(revenue_list)

        results.append({
            "gamma": gamma,
            "p_mean": p_mean, "p_std": p_std, "p_ci95": p_ci95, "p_list": p_list,
            "rr_mean": rr_mean, "rr_std": rr_std, "rr_ci95": rr_ci95, "rr_list": rr_list,
            "pim_mean": pim_mean, "pim_std": pim_std, "pim_ci95": pim_ci95, "pim_list": pim_list,
            "rev_mean": rev_mean, "rev_std": rev_std, "rev_ci95": rev_ci95, "rev_list": revenue_list,
        })
        if verbose:
            print(f"  -> summary: "
                  f"p* = {p_mean:6.3f} +/- {p_ci95:.3f}   "
                  f"pi_m* = {pim_mean:.4f} +/- {pim_ci95:.4f}   "
                  f"revenue* = {rev_mean:8.2f} +/- {rev_ci95:.2f}   "
                  f"revenue_rate* = {rr_mean:.4f} +/- {rr_ci95:.4f}")

    if verbose:
        print_summary_table(results)

    return results


def print_summary_table(results):
    """Print one consolidated table across all gamma values, after the
    per-repeat detail has already been printed by sweep_gamma()."""
    header = (f"\n{'gamma':>8} | {'p*':>10} | {'pi_m*':>12} | "
              f"{'revenue*':>14} | {'revenue_rate*':>16}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['gamma']:8.3f} | "
              f"{r['p_mean']:6.3f}+/-{r['p_ci95']:.3f} | "
              f"{r['pim_mean']:6.4f}+/-{r['pim_ci95']:.4f} | "
              f"{r['rev_mean']:8.2f}+/-{r['rev_ci95']:.2f} | "
              f"{r['rr_mean']:6.4f}+/-{r['rr_ci95']:.4f}")


def plot_gamma_sweep(results):
    gammas = [r["gamma"] for r in results]
    p_means = [r["p_mean"] for r in results]
    p_errs = [r["p_ci95"] for r in results]
    rr_means = [r["rr_mean"] for r in results]
    rr_errs = [r["rr_ci95"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].errorbar(gammas, p_means, yerr=p_errs, marker="o", capsize=4,
                      color="tab:blue")
    axes[0].set_xlabel("Abandonment rate gamma")
    axes[0].set_ylabel("Optimal price p*")
    axes[0].set_title("Optimal price vs. gamma")
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(gammas, rr_means, yerr=rr_errs, marker="o", capsize=4,
                      color="tab:red")
    axes[1].set_xlabel("Abandonment rate gamma")
    axes[1].set_ylabel("Max revenue rate")
    axes[1].set_title("Maximized revenue rate vs. gamma")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    lambda1 = 1
    lambda2 = 2
    q = 0.1
    termination_time = 1000
    p_low, p_high = 0, 100

    gamma_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    print("=== Sweeping gamma: golden-section search for p* at each gamma ===")
    results = sweep_gamma(
        gamma_list, lambda1, lambda2, q, termination_time,
        p_low=p_low, p_high=p_high,
        n_repeats=5,     # bump up for tighter CIs
        tol=1.0,
        n_reps=10, eval_reps=20,
        use_crn=True, crn_reps=20,
        verbose=True)

    plot_gamma_sweep(results)