import random
import matplotlib.pyplot as plt

# 本脚本单独用来诊断"方向2"加权决策模型里 (pi_m1, pi_m) 联合不动点
# 是否真的收敛，以及收敛得快不快、有没有震荡。
# 不依赖 weighted_decision_model.py，这里重新写了一份带"历史记录"的
# fixed_point_solve_v2_diag，其余仿真逻辑与 simulate_v2 完全一致。


def generate_exp(rate):
    return random.expovariate(rate)


def simulate_v2(pi_m1, pi_m, p, lambda1, lambda2, q, gamma, w, termination_time):
    Q1, Q2 = 0, 0
    event_calendar = [generate_exp(lambda1), generate_exp(lambda2), termination_time]
    direction = [0, 0, 0]
    t = 0.0
    joined1, joined2 = 0, 0
    abandoned1, abandoned2 = 0, 0

    pi_eff = w * pi_m1 + (1 - w) * pi_m

    while t < termination_time:
        tn = min(event_calendar)
        col = event_calendar.index(tn)
        t = tn

        if col == 0:
            u = random.uniform(0, 100)
            if u * pi_eff > p:
                joined1 += 1
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
            joined2 += 1
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
            if direction[col] == 1:
                Q1 -= 1
                abandoned1 += 1
            else:
                Q2 -= 1
                abandoned2 += 1
            del event_calendar[col]
            del direction[col]

    return joined1, joined2, abandoned1, abandoned2


# --- 带历史记录的联合不动点求解：每一轮都记录 (pi_m1, pi_m, diff1, diff) ---
def fixed_point_solve_v2_diag(p, lambda1, lambda2, q, gamma, w, termination_time,
                               tol=1e-2, max_iter=100, n_reps=20, damping=1.0):
    pi_m1, pi_m = 1.0, 1.0
    history = [(0, pi_m1, pi_m, None, None)]  # 第0轮：初始值，还没有 diff

    for k in range(1, max_iter + 1):
        j1_sum, j2_sum, a1_sum, a2_sum = 0, 0, 0, 0
        for _ in range(n_reps):
            j1, j2, a1, a2 = simulate_v2(
                pi_m1, pi_m, p, lambda1, lambda2, q, gamma, w, termination_time)
            j1_sum += j1
            j2_sum += j2
            a1_sum += a1
            a2_sum += a2

        if j1_sum == 0:
            print(f"  [ratio 对应的 p={p}] 第{k}轮: 雇主无人加入，提前终止")
            break

        total_joined = j1_sum + j2_sum
        total_abandoned = a1_sum + a2_sum

        pi_m1_new = 1 - a1_sum / j1_sum
        pi_m_new = 1 - total_abandoned / total_joined

        diff1 = abs(pi_m1_new - pi_m1)
        diff = abs(pi_m_new - pi_m)

        pi_m1 = pi_m1 + damping * (pi_m1_new - pi_m1)
        pi_m = pi_m + damping * (pi_m_new - pi_m)

        history.append((k, pi_m1, pi_m, diff1, diff))

        if diff1 < tol and diff < tol:
            print(f"  [p={p}] 收敛于第 {k} 轮 (diff1={diff1:.6f}, diff={diff:.6f})")
            
        if  k==max_iter:
                    break

    return pi_m1, pi_m, history


# ---------------------------------------------------------------------
# 参数：挑几个有代表性的 (lambda1, lambda2) 组合做收敛性检验，
# 覆盖低/中/高失衡程度，价格先用一个固定的合理值（不跑黄金分割，
# 只是单纯检验"给定 p，联合不动点会不会收敛"这件事本身）
# ---------------------------------------------------------------------
q = 0.1
gamma = 0.2
termination_time = 1000
n_reps = 20
w = 0.8
p_test = 15   # 固定一个测试价格；如果想测试极端价格下是否也能收敛，可以另外加

test_cases = [
    ("ratio=0.5 (more employers)", 1, 0.5),
    ("ratio=2 (base case)", 1, 2),
    ("ratio=8 (much more employees)", 1, 8),
]

plt.figure(figsize=(12, 8))

for idx, (label, lambda1, lambda2) in enumerate(test_cases):
    print(f"\n=== {label}: lambda1={lambda1}, lambda2={lambda2} ===")
    pi_m1_final, pi_m_final, history = fixed_point_solve_v2_diag(
        p_test, lambda1, lambda2, q, gamma, w, termination_time,
        tol=1e-2, max_iter=100, n_reps=n_reps)

    iters = [h[0] for h in history]
    pi_m1_hist = [h[1] for h in history]
    pi_m_hist = [h[2] for h in history]
    diff1_hist = [h[3] for h in history[1:]]  # 跳过第0轮（没有diff）
    diff_hist = [h[4] for h in history[1:]]

    # 子图1：pi_m1、pi_m 的轨迹（是否平稳收敛到一个值，还是震荡）
    plt.subplot(2, 3, idx + 1)
    plt.plot(iters, pi_m1_hist, marker='o', markersize=3, label='pi_m1 (employer-side)')
    plt.plot(iters, pi_m_hist, marker='s', markersize=3, label='pi_m (platform-wide)')
    plt.xlabel("iteration")
    plt.ylabel("value")
    plt.title(f"{label}\ntrajectory")
    plt.legend(fontsize=8)
    plt.grid(True)

    # 子图2：diff 随迭代次数变化（log坐标），检验是否单调下降到 tol 以下
    plt.subplot(2, 3, idx + 4)
    plt.plot(iters[1:], diff1_hist, marker='o', markersize=3, label='diff1 (pi_m1)')
    plt.plot(iters[1:], diff_hist, marker='s', markersize=3, label='diff (pi_m)')
    plt.axhline(y=1e-2, color='gray', linestyle='--', linewidth=1, label='tol=1e-2')
    plt.yscale('log')
    plt.xlabel("iteration")
    plt.ylabel("|diff| (log scale)")
    plt.title("convergence diff")
    plt.legend(fontsize=8)
    plt.grid(True, which='both')

plt.tight_layout()
plt.savefig("pi_m1_pi_m_convergence_check.png", dpi=150)
plt.show()