"""
双端计费的联合不动点求解（fix method）—— 加权有效匹配率版本
------------------------------------------------------------
延续单侧定价部分的建模假设：雇主并非完全掌握自身专属匹配率 pi_m1，而是将其
与更易获取的平台整体匹配率 pi_m 按权重加权，构成决策依据的"有效匹配率"。
双端定价下，雇员侧的准入决策沿用同一信息结构，以保持全文模型设定一致：

  雇主: u1 * (w1 * pi_m1 + (1-w1) * pi_m) > p1      u1 ~ U(0, 100)
  雇员: u2 * (w2 * pi_m2 + (1-w2) * pi_m) > p2      u2 ~ U(0, u2_max)

w1 沿用单侧部分已确定的 0.8；w2 默认设为更小的 0.5——雇员是低风险、低投入
的一次性决策方，缺乏动机/渠道获取自身专属匹配率，因而更依赖平台整体口碑
这一易得信号。w1、w2 均可自行调整（取 w2=w1 即退化为完全对称的情形）。

pi_m1、pi_m2、pi_m 三者均由同一批仿真的加入/放弃统计联合定义，因此需要
联合迭代求解三者的自洽不动点。随机数生成器直接复用 GitHub 已有代码。
"""

import random
from goldenmethod import generate_exp  # 直接复用已有代码


# ---------- 1. 双端准入的仿真过程（准入依据改为加权有效匹配率） ----------
def simulate_two_sided(pi_m1, pi_m2, pi_m, w1, w2, p1, p2, lambda1, lambda2,
                        q, gamma, u2_max, termination_time):
    eff1 = w1 * pi_m1 + (1 - w1) * pi_m  # 雇主的有效匹配率
    eff2 = w2 * pi_m2 + (1 - w2) * pi_m  # 雇员的有效匹配率

    Q1 = Q2 = 0
    event_calendar = [generate_exp(lambda1), generate_exp(lambda2), termination_time]
    direction = [0, 0, 0]
    t = 0.0
    joined1 = joined2 = abandoned1 = abandoned2 = 0
    revenue = 0.0

    while t < termination_time:
        tn = min(event_calendar)
        col = event_calendar.index(tn)
        t = tn

        if col == 0:  # 雇主到达：u1 * eff1 > p1
            u1 = random.uniform(0, 100)
            if u1 * eff1 > p1:
                joined1 += 1
                revenue += p1
                uu = random.uniform(0, 1)
                if uu < (1 - q) ** Q2:
                    Q1 += 1
                    event_calendar.append(t + generate_exp(gamma))
                    direction.append(1)
                else:
                    Q2 -= 1
                    idx = random.choice([i for i, d in enumerate(direction) if d == 2])
                    del event_calendar[idx]; del direction[idx]
            event_calendar[0] = t + generate_exp(lambda1)

        elif col == 1:  # 雇员到达：u2 * eff2 > p2
            u2 = random.uniform(0, u2_max)
            if u2 * eff2 > p2:
                joined2 += 1
                revenue += p2
                uu = random.uniform(0, 1)
                if uu < (1 - q) ** Q1:
                    Q2 += 1
                    event_calendar.append(t + generate_exp(gamma))
                    direction.append(2)
                else:
                    Q1 -= 1
                    idx = random.choice([i for i, d in enumerate(direction) if d == 1])
                    del event_calendar[idx]; del direction[idx]
            event_calendar[1] = t + generate_exp(lambda2)

        elif col == 2:
            break

        else:  # 耐心耗尽，放弃离开
            if direction[col] == 1:
                Q1 -= 1; abandoned1 += 1
            else:
                Q2 -= 1; abandoned2 += 1
            del event_calendar[col]; del direction[col]

    return joined1, joined2, abandoned1, abandoned2, revenue, revenue / termination_time


# ---------- 2. 联合不动点迭代：pi_m1、pi_m2、pi_m 三者同时求解 ----------
def fixed_point_solve_two_sided(p1, p2, lambda1, lambda2, q, gamma, u2_max,
                                 termination_time, w1=0.8, w2=0.5,
                                 tol=1e-2, max_iter=100, n_reps=20, verbose=True):
    pi_m1 = pi_m2 = pi_m = 1.0
    for k in range(max_iter):
        j1 = j2 = a1 = a2 = 0
        for _ in range(n_reps):
            r = simulate_two_sided(pi_m1, pi_m2, pi_m, w1, w2, p1, p2,
                                    lambda1, lambda2, q, gamma, u2_max, termination_time)
            j1 += r[0]; j2 += r[1]; a1 += r[2]; a2 += r[3]

        if j1 == 0 or j2 == 0:
            if verbose:
                print(f"iter {k}: joined1={j1}, joined2={j2}，一侧无人加入，迭代终止")
            return pi_m1, pi_m2, pi_m, k + 1

        pi_m1_new = 1 - a1 / j1
        pi_m2_new = 1 - a2 / j2
        pi_m_new = 1 - (a1 + a2) / (j1 + j2)  # 全平台整体匹配率
        diff = max(abs(pi_m1_new - pi_m1), abs(pi_m2_new - pi_m2), abs(pi_m_new - pi_m))

        if verbose:
            print(f"iter {k:3d}: pi_m1={pi_m1:.4f}->{pi_m1_new:.4f}, "
                  f"pi_m2={pi_m2:.4f}->{pi_m2_new:.4f}, "
                  f"pi_m={pi_m:.4f}->{pi_m_new:.4f}, diff={diff:.6f}")

        pi_m1, pi_m2, pi_m = pi_m1_new, pi_m2_new, pi_m_new
        if diff < tol:
            if verbose:
                print(f"Converge to the {k}th iteration")
            break

    return pi_m1, pi_m2, pi_m, k + 1


# ---------- 3. 收敛后，报告关键指标 ----------
def report_two_sided(p1, p2, lambda1, lambda2, q, gamma, u2_max, termination_time,
                      w1=0.8, w2=0.5, tol=1e-3, max_iter=100, n_reps=20,
                      eval_reps=30, verbose=True):
    pi_m1, pi_m2, pi_m, n_iter = fixed_point_solve_two_sided(
        p1, p2, lambda1, lambda2, q, gamma, u2_max, termination_time,
        w1=w1, w2=w2, tol=tol, max_iter=max_iter, n_reps=n_reps, verbose=verbose)

    j1 = j2 = a1 = a2 = 0
    rev = rev_rate = 0.0
    for _ in range(eval_reps):
        r = simulate_two_sided(pi_m1, pi_m2, pi_m, w1, w2, p1, p2,
                                lambda1, lambda2, q, gamma, u2_max, termination_time)
        j1 += r[0]; j2 += r[1]; a1 += r[2]; a2 += r[3]
        rev += r[4]; rev_rate += r[5]

    result = {
        "pi_m1": pi_m1, "pi_m2": pi_m2, "pi_m": pi_m,   # 三个不动点值
        "avg_joined1": j1 / eval_reps,
        "avg_joined2": j2 / eval_reps,
        "avg_abandoned1": a1 / eval_reps,
        "avg_abandoned2": a2 / eval_reps,
        "avg_revenue": rev / eval_reps,
        "avg_revenue_rate": rev_rate / eval_reps,
        "avg_revenue_rate1": (j1 / eval_reps) * p1 / termination_time,  # 雇主侧收益率
        "avg_revenue_rate2": (j2 / eval_reps) * p2 / termination_time,  # 雇员侧收益率
    }
    return result


if __name__ == "__main__":
    lambda1, lambda2, q, gamma = 1, 2, 0.1, 0.2
    u2_max = 15
    termination_time = 1000
    p1, p2 = 15, 1

    result = report_two_sided(p1, p2, lambda1, lambda2, q, gamma, u2_max,
                               termination_time, w1=0.8, w2=0.7,
                               n_reps=20, eval_reps=30)

    print("\n=== 关键量汇总 ===")
    for k, v in result.items():
        print(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")