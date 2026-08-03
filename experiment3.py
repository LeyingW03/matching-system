import random
import numpy as np
import matplotlib.pyplot as plt

# 复用仓库里已有的 fixed_point_solve，保证每个 q 对应的 pi_m 是真实均衡值
# 而不是像上一版诊断那样人为固定成 0.5
from goldenmethod import fixed_point_solve

# ---------------------------------------------------------------------
# 生成指数分布随机数（和仓库里 simulate() 的写法保持一致）
# ---------------------------------------------------------------------
def generate_exp(rate):
    return random.expovariate(rate)


# ---------------------------------------------------------------------
# simulate_diag：在仓库原版 simulate() 的基础上，额外记录：
#   - avg_Q1 / avg_Q2 ：整段仿真时间窗口内的时间加权平均队长
#   - employer_instant_match / employer_wait ：雇主"到达即匹配成功" vs "进入排队"次数
#   - employee_instant_match / employee_wait ：雇员"到达即匹配成功" vs "进入排队"次数
# 其余状态转移逻辑（谁跟谁匹配、放弃机制等）完全照搬仓库原版 simulate()，
# 只是多做了几个计数器，不改变系统本身的动态。
# ---------------------------------------------------------------------
def simulate_diag(pi_m, p, lambda1, lambda2, q, gamma, termination_time):
    Q1 = 0
    Q2 = 0

    event_calendar = [
        generate_exp(lambda1),
        generate_exp(lambda2),
        termination_time,
    ]
    direction = [0, 0, 0]

    t = 0.0
    total_joined1 = 0
    total_joined2 = 0
    total_abandoned = 0

    # ---- 新增的诊断量 ----
    time_weighted_Q1 = 0.0
    time_weighted_Q2 = 0.0
    employer_instant_match = 0
    employer_wait = 0
    employee_instant_match = 0
    employee_wait = 0
    last_t = 0.0

    while t < termination_time:
        tn = min(event_calendar)
        col = event_calendar.index(tn)

        # 累积上一段时间里 Q1、Q2 对时间的积分，用于算时间加权平均队长
        time_weighted_Q1 += Q1 * (tn - last_t)
        time_weighted_Q2 += Q2 * (tn - last_t)
        last_t = tn
        t = tn

        # ---------------- 雇主(employer)一侧到达 ----------------
        if col == 0:
            u = random.uniform(0, 100)
            if u * pi_m > p:
                total_joined1 += 1
                uu = random.uniform(0, 1)

                if uu < (1 - q) ** Q2:          # 未匹配成功 -> 排队等待
                    Q1 += 1
                    employer_wait += 1
                    event_calendar.append(t + generate_exp(gamma))
                    direction.append(1)
                else:                            # 匹配成功 -> 立刻带走 Q2 里一个雇员
                    Q2 -= 1
                    employer_instant_match += 1
                    q2_indices = [i for i, d in enumerate(direction) if d == 2]
                    leave_idx = random.choice(q2_indices)
                    del event_calendar[leave_idx]
                    del direction[leave_idx]

            event_calendar[0] = t + generate_exp(lambda1)

        # ---------------- 雇员(employee)一侧到达 ----------------
        elif col == 1:
            total_joined2 += 1
            uu = random.uniform(0, 1)

            if uu < (1 - q) ** Q1:              # 未匹配成功 -> 排队等待
                Q2 += 1
                employee_wait += 1
                event_calendar.append(t + generate_exp(gamma))
                direction.append(2)
            else:                                # 匹配成功 -> 立刻带走 Q1 里一个雇主
                Q1 -= 1
                employee_instant_match += 1
                q1_indices = [i for i, d in enumerate(direction) if d == 1]
                leave_idx = random.choice(q1_indices)
                del event_calendar[leave_idx]
                del direction[leave_idx]

            event_calendar[1] = t + generate_exp(lambda2)

        # ---------------- 仿真终止 ----------------
        elif col == 2:
            break

        # ---------------- 放弃(abandon)事件 ----------------
        else:
            total_abandoned += 1
            if direction[col] == 1:
                Q1 -= 1
            else:
                Q2 -= 1
            del event_calendar[col]
            del direction[col]

    avg_Q1 = time_weighted_Q1 / termination_time
    avg_Q2 = time_weighted_Q2 / termination_time
    total_joined = total_joined1 + total_joined2

    return dict(
        avg_Q1=avg_Q1,
        avg_Q2=avg_Q2,
        employer_instant_match=employer_instant_match,
        employer_wait=employer_wait,
        employee_instant_match=employee_instant_match,
        employee_wait=employee_wait,
        total_joined=total_joined,
        total_abandoned=total_abandoned,
    )


# ---------------------------------------------------------------------
# base case 参数（除 q 外均固定不变）
# ---------------------------------------------------------------------
lambda1 = 1
lambda2 = 2
gamma = 0.2
termination_time = 1000
p_fixed = 15          # 固定价格，避免价格搜索噪声混入诊断结果

n_reps_fp = 100         # fixed_point_solve 内部，每轮迭代的重复仿真次数
n_reps_diag = 100        # 诊断阶段，对每个 q 独立重复仿真的次数

q_values = np.arange(0.1, 1.0, 0.05)

pi_m_list = []
avg_Q1_list = []
avg_Q2_list = []
employee_match_rate_list = []
employer_match_rate_list = []

for q in q_values:
    # 关键：每个 q 都先用仓库自带的 fixed_point_solve 求出真实均衡 pi_m，
    # 而不是像上一版诊断那样所有 q 共用同一个固定值。
    pi_m_eq, _ = fixed_point_solve(
        p_fixed, lambda1, lambda2, q, gamma, termination_time,
        n_reps=n_reps_fp, verbose=False)
    pi_m_list.append(pi_m_eq)

    results = [
        simulate_diag(pi_m_eq, p_fixed, lambda1, lambda2, q, gamma, termination_time)
        for _ in range(n_reps_diag)
    ]

    avg_Q1_list.append(np.mean([r["avg_Q1"] for r in results]))
    avg_Q2_list.append(np.mean([r["avg_Q2"] for r in results]))

    emp_inst = sum(r["employee_instant_match"] for r in results)
    emp_wait = sum(r["employee_wait"] for r in results)
    employee_match_rate_list.append(emp_inst / (emp_inst + emp_wait))

    empr_inst = sum(r["employer_instant_match"] for r in results)
    empr_wait = sum(r["employer_wait"] for r in results)
    employer_match_rate_list.append(empr_inst / (empr_inst + empr_wait))

    print(f"q={q:.2f} -> pi_m_eq={pi_m_eq:.4f}, avg_Q1={avg_Q1_list[-1]:.3f}, "
          f"avg_Q2={avg_Q2_list[-1]:.3f}, employee_match_rate={employee_match_rate_list[-1]:.4f}")

# ---------------------------------------------------------------------
# 画图：均衡 pi_m、队长、雇员即时匹配率 随 q 变化
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(q_values, pi_m_list, marker='o', color='purple')
axes[0].set_xlabel("q")
axes[0].set_ylabel("equilibrium pi_m")
axes[0].set_title(f"Equilibrium pi_m vs q (p={p_fixed})")
axes[0].grid(True)

axes[1].plot(q_values, avg_Q1_list, marker='o', label='avg Q1 (employer queue)')
axes[1].plot(q_values, avg_Q2_list, marker='o', label='avg Q2 (employee queue)')
axes[1].set_xlabel("q")
axes[1].set_ylabel("average queue length")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(q_values, employee_match_rate_list, marker='o', color='tab:green',
             label='employee instant-match rate')
axes[2].plot(q_values, employer_match_rate_list, marker='o', color='tab:blue',
             label='employer instant-match rate')
axes[2].set_xlabel("q")
axes[2].set_ylabel("instant-match rate")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("queue_diagnostics_equilibrium_pi_m.png", dpi=150)
plt.show()