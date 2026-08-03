import numpy as np
import matplotlib.pyplot as plt

# 直接复用 GitHub 仓库里 goldenmethod.py 中已经写好的方法：
# golden_section_search 内部会对每个候选价格 p 先调用 fixed_point_solve
# 求出 pi_m*，再据此计算 revenue_rate，然后用黄金分割法比较、收缩区间。
# 所以这里不需要重写 fixed_point_solve / golden_section_search，直接 import 即可。
from goldenmethod import golden_section_search

# ---------------------------------------------------------------------
# base case 参数（题目要求：除 q 外，其它量都固定不变）
# ---------------------------------------------------------------------
lambda1 = 1
lambda2 = 1.5
gamma = 0.2
termination_time = 1000
n_reps = 20        # fixed_point_solve 内部每次迭代的重复仿真次数
eval_reps = 20      # 收敛后统计 revenue_rate 的重复仿真次数

p_low, p_high = 0, 100   # 黄金分割搜索的价格区间，可按需调整
tol = 1            # 价格分辨率（收敛到约 1 个单位）

# q 的扫描范围（包含 base case 里的 q=0.3）
q_values = np.arange(0.1, 1.0, 0.1)

# ---------------------------------------------------------------------
# 对每个 q：调用仓库自带的 golden_section_search 求最优价格 p*
# ---------------------------------------------------------------------
p_star_list = []
revenue_rate_star_list = []

for q in q_values:
    p_star, pi_m_star, revenue_star, revenue_rate_star, _ = golden_section_search(
        lambda1, lambda2, q, gamma, termination_time,
        p_low=p_low, p_high=p_high, tol=tol,
        n_reps=n_reps, eval_reps=eval_reps, verbose=False)
    p_star_list.append(p_star)
    revenue_rate_star_list.append(revenue_rate_star)
    print(f"q={q:.2f} -> p*={p_star:.3f}, pi_m*={pi_m_star:.4f}, "
          f"revenue_rate*={revenue_rate_star:.4f}")

# ---------------------------------------------------------------------
# 画图 1：p* 随 q 变化
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(q_values, p_star_list, marker='o')
plt.xlabel("q (single match probability)")
plt.ylabel("optimal price p*")
plt.title("Optimal price p* vs matching probability q\n"
          f"(lambda1={lambda1}, lambda2={lambda2}, gamma={gamma})")
plt.grid(True)
plt.tight_layout()
plt.savefig("pstar_vs_q.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 画图 2：最优价格 p* 处对应的 revenue_rate* 随 q 变化
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(q_values, revenue_rate_star_list, marker='o', color='tab:red')
plt.xlabel("q (single match probability)")
plt.ylabel("optimal revenue rate (at p*)")
plt.title("Optimal revenue rate vs matching probability q\n"
          f"(lambda1={lambda1}, lambda2={lambda2}, gamma={gamma})")
plt.grid(True)
plt.tight_layout()
plt.savefig("revenue_rate_star_vs_q.png", dpi=150)
plt.show()
