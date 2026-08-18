"""
Nelder-Mead 多起点稳健性检验（覆盖多个供需比 lambda2）
--------------------------------------------------------
固定其他参数（lambda1, q, gamma, u2_max, w1, w2），对每个待检验的 lambda2 值，
分别用多个分散的初始价格对跑 Nelder-Mead，观察收敛到的 (p1*,p2*,revenue_rate*)
是否一致。

针对 scipy 默认 Nelder-Mead "原地打转"的已知问题（默认初始单纯形只按 x0 的
~5% 扰动构造，且 p1、p2 两个维度量纲差异大时问题更明显），做了三处改进：
  1. 手动指定初始单纯形，按 p1、p2 各自的量纲给出合适的扰动步长
  2. 打开 adaptive=True，让反射/扩张/收缩幅度按问题维度自动调整
  3. 周期性重置单纯形：收敛后以当前解为新起点重新构造单纯形再跑一轮，
     重复 n_restarts 次，降低早熟收敛/卡在局部最优的风险
"""

import numpy as np
from scipy.optimize import minimize
from two_sided_fixed_point import report_two_sided  # 直接复用已有代码

lambda1, q, gamma, u2_max = 1, 0.1, 0.2, 15
w1, w2 = 0.8, 0.7
termination_time = 300
n_reps, eval_reps = 10, 10 # 用较大重复数压低噪声，让不同起点的对比更可信

lambda2_to_check = [1,1.5,1.7,1.8,1.9,2,2.1,2.3,2.5,3,4,6,10,15,20]                  # 待检验的供需比，重点覆盖临界区间附近
start_points =   [(20,1)]

STEP_P1, STEP_P2 = 15.0, 3.0  # 初始单纯形的扰动步长，按 p1(~100)、p2(~15) 各自量纲给出，
                              # 而不是用 scipy 默认按 x0 等比例扰动（会导致 p2 方向步长过小）
N_RESTARTS = 3            # 单纯形重置次数：收敛后以当前解为中心重新构造单纯形再搜一轮

PENALTY = 1e6  # p1>=100 或 p2>=u2_max 时的惩罚值：此时准入条件恒为假，收益率恒为0，
               # 但 fixed_point_solve_two_sided 可能因提前退出返回未经验证的乐观 pi_m，
               # 直接跳过仿真、给一个明确的差值，避免 Nelder-Mead 被这类死区误导


def neg_revenue_rate(p, lambda2):
    p1, p2 = max(p[0], 0.0), max(p[1], 0.0)
    if p1 >= 100 or p2 >= u2_max:
        return PENALTY
    result = report_two_sided(p1, p2, lambda1, lambda2, q, gamma, u2_max,
                               termination_time, w1=w1, w2=w2,
                               n_reps=n_reps, eval_reps=eval_reps, verbose=False)
    return -result["avg_revenue_rate"]


def build_initial_simplex(center, step1=STEP_P1, step2=STEP_P2):
    """以 center 为基准顶点，按各维度步长构造一个 3 顶点的初始单纯形（2 变量问题）。"""
    c1, c2 = center
    return np.array([[c1, c2], [c1 + step1, c2], [c1, c2 + step2]])


def run_nelder_mead(p0, lambda2, n_restarts=N_RESTARTS):
    """跑一次 Nelder-Mead，然后以当前解为中心重置单纯形、重复 n_restarts 次。"""
    for _ in range(n_restarts):
        simplex = build_initial_simplex(p0)
        res = minimize(neg_revenue_rate, x0=p0, args=(lambda2,), method="Nelder-Mead",
                        options={"xatol": 0.5, "fatol": 1e-3, "maxiter": 60,
                                 "adaptive": True, "initial_simplex": simplex})
        p0 = (max(res.x[0], 0.0), max(res.x[1], 0.0))
    return p0[0], p0[1], -res.fun


for lambda2 in lambda2_to_check:
    print(f"\n=== lambda2={lambda2} (ratio={lambda2/lambda1:.2f}) ===", flush=True)
    for p0 in start_points:
        p1_star, p2_star, revenue_rate = run_nelder_mead(p0, lambda2)
        print(f"  start={p0}: -> p1*={p1_star:6.2f}, p2*={p2_star:5.2f}, "
              f"revenue_rate*={revenue_rate:.4f}", flush=True)