"""
单侧定价的供需比扫描（与双侧模型口径一致的版本）
--------------------------------------------------
不再使用 goldenmethod.py 里未加权的旧版 pi_m 模型，而是直接复用
two_sided_fixed_point_weighted.py 中的加权有效匹配率模型，把 p2 固定为 0。
此时雇员的准入条件 u2*eff2 > 0 对几乎所有 u2>0 恒成立（因为 u2~U(0,u2_max)，
u2=0 概率为零），等价于"雇员无条件加入"——正是原始单侧模型的假设。
这样单侧曲线与双侧曲线来自同一套模型，只是 p2 是否被优化的区别，
"该不该向雇员收费"的对比才是干净的、可以互相归因的。

固定 lambda1=1, q=0.1, gamma=0.2, u2_max=15, w1=0.8, w2=0.5；lambda2 用与
two_side_sweep.py 相同的扫描点。
"""

import csv
from scipy.optimize import minimize_scalar
from two_sided_fixed_point import report_two_sided  # 直接复用已有代码

lambda1, q, gamma, u2_max = 1, 0.1, 0.2, 15
w1, w2 = 0.8, 0.7
termination_time =300
lambda2_list = [2,4,6,10,15,20]  # 与 two_side_sweep.py 保持一致

n_reps, eval_reps = 20, 20  # 扫描期间用较小重复数控制耗时，最终点会再加密评

def neg_revenue_rate(p1, lambda2):
    p1 = max(p1, 0.0)
    result = report_two_sided(p1, 0.0, lambda1, lambda2, q, gamma, u2_max,
                               termination_time, w1=w1, w2=w2,
                               n_reps=n_reps, eval_reps=eval_reps, verbose=False)
    return -result["avg_revenue_rate"]


results = []
for lambda2 in lambda2_list:
    res = minimize_scalar(neg_revenue_rate, args=(lambda2,), method="bounded",
                           bounds=(0, 100), options={"xatol": 0.5})
    p1_star = max(res.x, 0.0)

    final = report_two_sided(p1_star, 0.0, lambda1, lambda2, q, gamma, u2_max,
                              termination_time, w1=w1, w2=w2,
                              n_reps=n_reps, eval_reps=eval_reps * 2, verbose=False)

    results.append({"lambda2": lambda2, "ratio": lambda2 / lambda1,
                     "p1_star": p1_star, "revenue_rate": final["avg_revenue_rate"]})
    print(f"lambda2={lambda2:5.2f} (ratio={lambda2/lambda1:5.2f}): "
          f"p1*={p1_star:6.2f}, revenue_rate*={final['avg_revenue_rate']:.4f}")

with open("single_side_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["lambda2", "ratio", "p1_star", "revenue_rate"])
    writer.writeheader()
    writer.writerows(results)

print("\n已保存至 single_side_results.csv")

print("\n=== 全部结果汇总 ===")
print(f"{'lambda2':>8} {'ratio':>8} {'p1_star':>10} {'revenue_rate':>14}")
for r in results:
    print(f"{r['lambda2']:>8.2f} {r['ratio']:>8.2f} {r['p1_star']:>10.2f} {r['revenue_rate']:>14.4f}")