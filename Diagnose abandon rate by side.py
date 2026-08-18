import random
import numpy as np
import matplotlib.pyplot as plt

# 复用 GitHub 仓库 goldenmethod.py 里已经写好的方法：
# - golden_section_search: 用来给每个 (lambda1, lambda2) 先找到 p*（和之前完全一样，不改动）
# 这里不复用 simulate() / fixed_point_solve()，因为它们把雇主、雇员的
# abandoned 计数混在一起（total_abandoned），没有暴露分侧数据。
# 下面 simulate_diag() 是 simulate() 的诊断版拷贝：只多做一件事——
# 按 direction（1=雇主排队, 2=雇员排队）把 abandoned 分开计数，
# 决策规则（u*pi_m > p）和匹配/放弃机制完全不变,不影响主结果。
from goldenmethod import golden_section_search


def generate_exp(rate):
    return random.expovariate(rate)


# --- simulate() 的诊断版拷贝：只新增分侧统计，不改变任何决策/匹配逻辑 ---
def simulate_diag(pi_m, p, lambda1, lambda2, q, gamma, termination_time):
    Q1 = 0
    Q2 = 0
    event_calendar = [generate_exp(lambda1), generate_exp(lambda2), termination_time]
    direction = [0, 0, 0]
    t = 0.0

    joined1, joined2 = 0, 0          # 雇主 / 雇员 分别加入人数
    abandoned1, abandoned2 = 0, 0    # 雇主 / 雇员 分别放弃人数

    while t < termination_time:
        tn = min(event_calendar)
        col = event_calendar.index(tn)
        t = tn

        if col == 0:  # 雇主到达
            u = random.uniform(0, 100)
            if u * pi_m > p:
                joined1 += 1
                uu = random.uniform(0, 1)
                if uu < (1 - q) ** Q2:  # 未即时配对，进入 Q1 排队
                    Q1 += 1
                    event_calendar.append(t + generate_exp(gamma))
                    direction.append(1)
                else:  # 即时配对成功
                    Q2 -= 1
                    q2_indices = [i for i, d in enumerate(direction) if d == 2]
                    leave_idx = random.choice(q2_indices)
                    del event_calendar[leave_idx]
                    del direction[leave_idx]
            event_calendar[0] = t + generate_exp(lambda1)

        elif col == 1:  # 雇员到达
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

        elif col == 2:  # 终止
            break

        else:  # 耐心耗尽，放弃
            if direction[col] == 1:
                Q1 -= 1
                abandoned1 += 1
            else:
                Q2 -= 1
                abandoned2 += 1
            del event_calendar[col]
            del direction[col]

    return joined1, joined2, abandoned1, abandoned2


# --- fixed_point_solve() 的诊断版：迭代方式和收敛判据与原版完全一致
#     （仍然用 total_abandoned/total_joined 更新 pi_m，不改变决策机制），
#     只是收敛后额外汇报分侧的放弃率，供诊断使用 ---
def fixed_point_solve_diag(p, lambda1, lambda2, q, gamma, termination_time,
                            tol=1e-4, max_iter=100, n_reps=20):
    pi_m = 1.0
    for k in range(max_iter):
        j1_sum, j2_sum, a1_sum, a2_sum = 0, 0, 0, 0
        for _ in range(n_reps):
            j1, j2, a1, a2 = simulate_diag(pi_m, p, lambda1, lambda2, q, gamma, termination_time)
            j1_sum += j1
            j2_sum += j2
            a1_sum += a1
            a2_sum += a2

        total_joined = j1_sum + j2_sum
        total_abandoned = a1_sum + a2_sum
        if total_joined == 0:
            break

        pi_m_new = 1 - total_abandoned / total_joined
        diff = abs(pi_m_new - pi_m)
        pi_m = pi_m_new
        if diff < tol:
            break

    # 收敛后，在最终 pi_m 下再跑 n_reps 次，专门用来估计稳定的分侧放弃率
    # （避免直接用收敛过程最后一轮的样本，样本量偏小、噪声较大）
    j1_sum, j2_sum, a1_sum, a2_sum = 0, 0, 0, 0
    for _ in range(n_reps):
        j1, j2, a1, a2 = simulate_diag(pi_m, p, lambda1, lambda2, q, gamma, termination_time)
        j1_sum += j1
        j2_sum += j2
        a1_sum += a1
        a2_sum += a2

    employer_abandon_rate = a1_sum / j1_sum if j1_sum > 0 else float('nan')
    employee_abandon_rate = a2_sum / j2_sum if j2_sum > 0 else float('nan')
    overall_abandon_rate = (a1_sum + a2_sum) / (j1_sum + j2_sum)

    return pi_m, employer_abandon_rate, employee_abandon_rate, overall_abandon_rate


# ---------------------------------------------------------------------
# base case 参数，和之前的实验保持一致
# ---------------------------------------------------------------------
q = 0.1
gamma = 0.2
termination_time = 1000
n_reps = 20
eval_reps = 20
p_low, p_high = 0, 100
tol = 1

lambda1 = 1
ratio_values = [0.5, 0.75, 1, 1.5, 2, 3, 5, 8]
lambda2_values = [lambda1 * r for r in ratio_values]

# ---------------------------------------------------------------------
# 对每个 ratio：
#   1) 先用仓库自带的 golden_section_search 找到 p*（和之前完全一致）
#   2) 在这个 p* 下，用诊断版 fixed point 拆出雇主/雇员分侧放弃率
# ---------------------------------------------------------------------
p_star_list = []
employer_abandon_list = []
employee_abandon_list = []
overall_abandon_list = []

for ratio, lambda2 in zip(ratio_values, lambda2_values):
    p_star, _, _, _, _ = golden_section_search(
        lambda1, lambda2, q, gamma, termination_time,
        p_low=p_low, p_high=p_high, tol=tol,
        n_reps=n_reps, eval_reps=eval_reps, verbose=False)

    pi_m, emp_abandon, ee_abandon, overall_abandon = fixed_point_solve_diag(
        p_star, lambda1, lambda2, q, gamma, termination_time,
        tol=1e-4, max_iter=100, n_reps=n_reps)

    p_star_list.append(p_star)
    employer_abandon_list.append(emp_abandon)
    employee_abandon_list.append(ee_abandon)
    overall_abandon_list.append(overall_abandon)

    print(f"ratio={ratio:>4.2f} (lambda2={lambda2:>5.2f}) -> p*={p_star:6.3f}, "
          f"pi_m={pi_m:.4f} | employer_abandon={emp_abandon:.4f}, "
          f"employee_abandon={ee_abandon:.4f}, overall_abandon={overall_abandon:.4f}")

# ---------------------------------------------------------------------
# 画图：雇主放弃率 vs 雇员放弃率 vs 整体放弃率，随 ratio 变化
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5.5))
plt.plot(ratio_values, employer_abandon_list, marker='o', label='employer abandon rate (side 1)')
plt.plot(ratio_values, employee_abandon_list, marker='s', label='employee abandon rate (side 2)')
plt.plot(ratio_values, overall_abandon_list, marker='^', linestyle='--',
         color='gray', label='overall abandon rate (= 1 - pi_m)')
plt.axvline(x=2, color='gray', linestyle=':', linewidth=1, label='base case (ratio=2)')
plt.xlabel("ratio = lambda2 / lambda1")
plt.ylabel("abandonment rate")
plt.title("Employer vs employee abandonment rate at p*, across ratio\n"
          f"(lambda1={lambda1}, q={q}, gamma={gamma})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("abandon_rate_by_side_vs_ratio.png", dpi=150)
plt.show()