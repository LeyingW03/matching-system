
import numpy as np
import matplotlib.pyplot as plt

from revenuerate import simulate, fixed_point_solve  # reuse existing model


def evaluate_price_crn(p, lambda1, lambda2, q, gamma, termination_time,
                        crn_seeds, tol=1e-4, max_iter=100, n_reps=20, verbose=False):
    """
    Same contract as revenue.evaluate_price, but always driven by a fixed
    list of random seeds (crn_seeds) for the final evaluation stage, so
    that calls for different p reuse the same random streams.
    """
    import random

    pi_m_star, _ = fixed_point_solve(
        p, lambda1, lambda2, q, gamma, termination_time,
        tol=tol, max_iter=max_iter, n_reps=n_reps, verbose=verbose)

    revenue_rates = []
    for s in crn_seeds:
        random.seed(s)
        _, _, _, _, _, revenue_rate = simulate(
            pi_m_star, p, lambda1, lambda2, q, gamma, termination_time)
        revenue_rates.append(revenue_rate)
    random.seed()  # release the fixed stream afterwards

    revenue_rates = np.array(revenue_rates)
    return pi_m_star, float(revenue_rates.mean()), float(revenue_rates.std(ddof=1))


def sweep_prices_and_plot(p_values, lambda1, lambda2, q, gamma, termination_time,
                           n_reps=20, crn_reps=40, crn_seed=0,
                           save_path="revenue_rate_vs_price.png"):
    """
    Sweep p over p_values, estimate R(p) at each point with CRN, and plot
    the resulting revenue-rate curve with a shaded +/-1 std band and the
    empirical maximum marked, to visually support/refute unimodality.
    """
    import random

    rng = random.Random(crn_seed)
    crn_seeds = [rng.randrange(1, 2**31 - 1) for _ in range(crn_reps)]

    pi_m_list, mean_list, std_list = [], [], []
    for p in p_values:
        pi_m_star, mean_rr, std_rr = evaluate_price_crn(
            p, lambda1, lambda2, q, gamma, termination_time,
            crn_seeds=crn_seeds, n_reps=n_reps)
        pi_m_list.append(pi_m_star)
        mean_list.append(mean_rr)
        std_list.append(std_rr)
        print(f"p={p:7.2f} -> pi_m*={pi_m_star:.4f}, "
              f"avg_revenue_rate={mean_rr:9.4f} (± {std_rr:.4f}, n={crn_reps})")

    p_values = np.array(p_values, dtype=float)
    mean_arr = np.array(mean_list)
    std_arr = np.array(std_list)
    se_arr = std_arr / np.sqrt(crn_reps)  # standard error of the mean

    best_idx = int(np.argmax(mean_arr))
    best_p = p_values[best_idx]
    best_rate = mean_arr[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, mean_arr, marker="o", color="#2563eb", linewidth=2,
            label="Average Revenue Rate  $R(p)$")
    ax.fill_between(p_values, mean_arr - se_arr, mean_arr + se_arr,
                     color="#2563eb", alpha=0.18, label="±1 s.e. (CRN)")
    ax.scatter([best_p], [best_rate], color="#dc2626", zorder=5, s=60,
               label=f"Grid maximum at $p$={best_p:.1f}")
    ax.axvline(best_p, color="#dc2626", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("Admission Price $p$")
    ax.set_ylabel("Average Revenue Rate $R(p)$ (revenue per unit time)")
    ax.set_title("Revenue Rate vs. Price: Empirical Check of Unimodality")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\n图像已保存至: {save_path}")

    return p_values, mean_arr, se_arr, pi_m_list


if __name__ == "__main__":
    # Parameters consistent with the golden section search demo in
    # revenue.py (same lambda1/lambda2/q/gamma/termination_time and the
    # same price interval [0, 100] used as the search bounds), so the
    # unimodality check directly supports the interval actually searched.
    lambda1 = 1
    lambda2 = 2
    q = 0.1
    gamma = 0.2
    termination_time = 1000
    n_reps = 20     # replications used inside fixed_point_solve at each p
    crn_reps = 40   # replications (with common random numbers) used to estimate R(p)

    p_values = np.arange(0, 101, 5)  # coarse grid: 0, 5, 10, ..., 100

    sweep_prices_and_plot(
        p_values, lambda1, lambda2, q, gamma, termination_time,
        n_reps=n_reps, crn_reps=crn_reps, crn_seed=0,
        save_path=r"C:\Users\s2829716\Documents\revenue_rate_vs_price.png")