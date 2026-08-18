import numpy as np
import matplotlib.pyplot as plt

# 直接复用 GitHub 仓库里 goldenmethod.py 中已经写好的方法：
# golden_section_search 内部会对每个候选价格 p 先调用 fixed_point_solve
# 求出 pi_m*（黄金分割法调用 fixed point 法），再据此计算 revenue_rate，
# 然后用黄金分割法比较、收缩区间，找出最优价格 p*。
# 所以这里不需要重写 fixed_point_solve / golden_section_search，直接 import 即可。
# 使用前请确保本脚本与 goldenmethod.py 放在同一目录下。
from goldenmethod import golden_section_search

# ---------------------------------------------------------------------
# base case 参数（题目要求：除 (lambda1, lambda2) 外，其它量都固定不变）
# ---------------------------------------------------------------------
q = 0.1
gamma = 0.2
termination_time = 1000
n_reps = 20         # fixed_point_solve 内部每次迭代的重复仿真次数
eval_reps = 20       # 收敛后统计 revenue_rate 的重复仿真次数

p_low, p_high = 0, 100   # 黄金分割搜索的价格区间，可按需调整
tol = 1              # 价格分辨率（收敛到约 1 个单位）

# ---------------------------------------------------------------------
# lambda1 固定为 base case 的值（=1），只让 lambda2 随比例 ratio 变化
# 即固定"整体拥堵程度" lambda1/gamma 不变，只改变"供需失衡比例" ratio = lambda2/lambda1
# ---------------------------------------------------------------------
lambda1 = 1

# ratio 取值：覆盖 ratio<1（雇主相对稀缺，对照组）、ratio=1（均衡基准）、
# ratio>1（雇员相对富余，贴合现实中"求职者多于岗位"的主流情况，重点密集取点）
ratio_values = [0.5, 0.75, 1, 1.5, 2, 3, 5, 8]
lambda2_values = [lambda1 * r for r in ratio_values]

# ---------------------------------------------------------------------
# 对每个 (lambda1, lambda2)：调用仓库自带的 golden_section_search 求最优价格 p*
# ---------------------------------------------------------------------
p_star_list = []
revenue_rate_star_list = []

for ratio, lambda2 in zip(ratio_values, lambda2_values):
    p_star, pi_m_star, revenue_star, revenue_rate_star, _ = golden_section_search(
        lambda1, lambda2, q, gamma, termination_time,
        p_low=p_low, p_high=p_high, tol=tol,
        n_reps=n_reps, eval_reps=eval_reps, verbose=False)
    p_star_list.append(p_star)
    revenue_rate_star_list.append(revenue_rate_star)
    print(f"ratio={ratio:.2f} (lambda1={lambda1}, lambda2={lambda2:.2f}) -> "
          f"p*={p_star:.3f}, pi_m*={pi_m_star:.4f}, revenue_rate*={revenue_rate_star:.4f}")

# ---------------------------------------------------------------------
# 画图 1：p* 随 lambda2/lambda1 比例变化
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(ratio_values, p_star_list, marker='o')
plt.axvline(x=2, color='gray', linestyle='--', linewidth=1, label='base case (ratio=2)')
plt.xlabel("ratio = lambda2 / lambda1")
plt.ylabel("optimal price p*")
plt.title("Optimal price p* vs supply-demand ratio (lambda2/lambda1)\n"
          f"(lambda1={lambda1}, q={q}, gamma={gamma})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pstar_vs_lambda_ratio.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 画图 2：最优价格 p* 处对应的 revenue_rate* 随 lambda2/lambda1 比例变化
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(ratio_values, revenue_rate_star_list, marker='o', color='tab:red')
plt.axvline(x=2, color='gray', linestyle='--', linewidth=1, label='base case (ratio=2)')
plt.xlabel("ratio = lambda2 / lambda1")
plt.ylabel("optimal revenue rate (at p*)")
plt.title("Optimal revenue rate vs supply-demand ratio (lambda2/lambda1)\n"
          f"(lambda1={lambda1}, q={q}, gamma={gamma})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("revenue_rate_star_vs_lambda_ratio.png", dpi=150)
plt.show()