import random
import numpy as np

def generate_exp(rate):
    return random.expovariate(rate)

def simulate(pi_m, p, lambda1, lambda2, q, gamma, termination_time):
    Q1 = 0
    Q2 = 0
    event_calendar = [generate_exp(lambda1), generate_exp(lambda2), termination_time]
    direction = [0, 0, 0]
    t = 0.0
    total_joined1 = 0
    total_joined2 = 0
    total_abandoned = 0
    total_revenue = 0.0

    while t < termination_time:
        tn = min(event_calendar)
        col = event_calendar.index(tn)
        t = tn

        if col == 0:
            u = random.uniform(0, 100)
            if u * pi_m > p:
                total_joined1 += 1
                total_revenue += p
                uu = random.uniform(0, 1)
                if uu < (1 - q) ** Q2:
                    Q1 += 1
                    event_calendar.append(t + generate_exp(gamma))
                    direction.append(1)
                else:
                    Q2 -= 1
                    q2_indices = [i for i, d in enumerate(direction) if d == 2]
                    leave_idx = random.choice(q2_indices)
                    del event_calendar[leave_idx]
                    del direction[leave_idx]
            event_calendar[0] = t + generate_exp(lambda1)

        elif col == 1:
            total_joined2 += 1
            uu = random.uniform(0, 1)
            if uu < (1 - q) ** Q1:
                Q2 += 1
                event_calendar.append(t + generate_exp(gamma))
                direction.append(2)
            else:
                Q1 -= 1
                q1_indices = [i for i, d in enumerate(direction) if d == 1]
                leave_idx = random.choice(q1_indices)
                del event_calendar[leave_idx]
                del direction[leave_idx]
            event_calendar[1] = t + generate_exp(lambda2)

        elif col == 2:
            break
        else:
            total_abandoned += 1
            if direction[col] == 1:
                Q1 -= 1
            else:
                Q2 -= 1
            del event_calendar[col]
            del direction[col]

    total_joined = total_joined1 + total_joined2
    revenue_rate = total_revenue / termination_time
    return total_joined, total_abandoned, Q1, Q2, total_revenue, revenue_rate


def g_map(pi_m_fixed, p, lambda1, lambda2, q, gamma, termination_time, n_reps=200):
    """在固定 pi_m_fixed 下（不迭代），跑 n_reps 次仿真，
    返回按不动点更新公式算出的平均 pi_m_new"""
    total_joined = 0
    total_abandoned = 0
    for _ in range(n_reps):
        joined, abandoned, _, _, _, _ = simulate(
            pi_m_fixed, p, lambda1, lambda2, q, gamma, termination_time)
        total_joined += joined
        total_abandoned += abandoned
    if total_joined == 0:
        return 0.0
    return 1 - total_abandoned / total_joined


lambda1, lambda2, q, gamma, termination_time = 1, 2, 0.1, 0.2, 1000

for p_test in [13, 15, 17]:
    print(f"\n===== p = {p_test} =====")
    pi_m_grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    results = []
    for pm in pi_m_grid:
        pm_new = g_map(pm, p_test, lambda1, lambda2, q, gamma, termination_time, n_reps=150)
        results.append(pm_new)
        diff = pm_new - pm
        marker = "<-- 交点附近" if abs(diff) < 0.03 else ""
        print(f"  pi_m={pm:.2f} -> g(pi_m)={pm_new:.4f}  (g-pi_m={diff:+.4f}) {marker}")