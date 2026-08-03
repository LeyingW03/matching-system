import numpy as np
import matplotlib.pyplot as plt
from goldenmethod import golden_section_search

lambda1, lambda2, gamma, termination_time = 1, 2, 0.2, 1000
n_reps, eval_reps = 40, 40
p_low, p_high, tol = 0, 100, 1.5

q_values = np.arange(0.1, 1.0, 0.05)
n_repeats = 8          # 关键新增：每个 q 独立重复跑多少次搜索

p_star_mean, p_star_lower, p_star_upper = [], [], []

for q in q_values:
    p_star_samples = []
    for _ in range(n_repeats):
        p_star, *_ = golden_section_search(
            lambda1, lambda2, q, gamma, termination_time,
            p_low=p_low, p_high=p_high, tol=tol,
            n_reps=n_reps, eval_reps=eval_reps, verbose=False)
        p_star_samples.append(p_star)

    p_star_samples = np.array(p_star_samples)
    p_star_mean.append(np.median(p_star_samples))          # 用中位数代替均值，更抗极端值
    lo, hi = np.percentile(p_star_samples, [2.5, 97.5])      # 95% 经验置信区间（百分位数法）
    p_star_lower.append(lo)
    p_star_upper.append(hi)

    print(f"q={q:.2f} -> p* median={p_star_mean[-1]:.2f}, "
          f"95% CI=[{lo:.2f}, {hi:.2f}], n={n_repeats}")

p_star_mean = np.array(p_star_mean)
p_star_lower = np.array(p_star_lower)
p_star_upper = np.array(p_star_upper)

plt.figure(figsize=(8, 5))
plt.plot(q_values, p_star_mean, marker='o', color='tab:blue', label='p* (median)')
plt.fill_between(q_values, p_star_lower, p_star_upper, alpha=0.25,
                  label='95% empirical CI (30 repeats)')
plt.xlabel("q")
plt.ylabel("optimal price p*")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pstar_vs_q_with_CI.png", dpi=150)
plt.show()