import random
import numpy as np
import matplotlib.pyplot as plt

def generate_exp(rate):
    return random.expovariate(rate)


def simulate(pi_m, p, lambda1, lambda2, q, gamma, termination_time):
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
    total_revenue = 0.0  # 平台总收益，仅雇主(employer)一方付费

    while t < termination_time:
        tn = min(event_calendar)
        col = event_calendar.index(tn)
        t = tn

        # 雇主(employer)一侧到达
        if col == 0:
            u = random.uniform(0, 100)
            if u * pi_m > p:
                total_joined1 += 1
                total_revenue += p  # 每有一个雇主加入，平台收益 +p
                uu = random.uniform(0, 1)

                if uu < (1 - q) ** Q2:  # 未匹配成功，进入队列等待
                    Q1 += 1
                    event_calendar.append(t + generate_exp(gamma))
                    direction.append(1)
                else:  # 匹配成功
                    Q2 -= 1
                    q2_indices = [i for i, d in enumerate(direction) if d == 2]
                    leave_idx = random.choice(q2_indices)
                    del event_calendar[leave_idx]
                    del direction[leave_idx]

            event_calendar[0] = t + generate_exp(lambda1)

        # 雇员(employee)一侧到达
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

        # 仿真终止
        elif col == 2:
            break

        # 放弃(abandon)事件
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


# ---------------------------------------------------------------------
# 2. 不动点求解（与 revenue.py 中的 fixed_point_solve 保持一致）
# ---------------------------------------------------------------------
def fixed_point_solve(p, lambda1, lambda2, q, gamma, termination_time,
                       tol=1e-4, max_iter=100, n_reps=20, verbose=False):
    pi_m = 0.1

    for k in range(max_iter):
        total_joined = 0
        total_abandoned = 0
        total_revenue = 0.0
        total_revenue_rate = 0.0

        for _ in range(n_reps):
            joined, abandoned, _, _, revenue, revenue_rate = simulate(
                pi_m, p, lambda1, lambda2, q, gamma, termination_time)
            total_joined += joined
            total_abandoned += abandoned
            total_revenue += revenue
            total_revenue_rate += revenue_rate

        # 价格过高，无人加入
        if total_joined == 0:
            if verbose:
                print(f"{k}: 价格过高，{n_reps}次重复仿真均无人加入，迭代终止")
            return pi_m, k + 1

        pi_m_new = 1 - total_abandoned / total_joined
        diff = abs(pi_m_new - pi_m)

        if verbose:
            avg_revenue = total_revenue / n_reps
            avg_revenue_rate = total_revenue_rate / n_reps
            print(f"iter {k:3d}: pi_m={pi_m_new:.6f}, revenue={avg_revenue:.2f}, "
                  f"revenue_rate={avg_revenue_rate:.4f}")

        pi_m = pi_m_new
        if diff < tol:
            break

    return pi_m, k + 1


# ---------------------------------------------------------------------
# 3. 给定价格 p，求解不动点 pi_m* 并统计平均收益 / 收益率
#    （对应 goldenmethod.py 中的 evaluate_price）
# ---------------------------------------------------------------------
def evaluate_price(p, lambda1, lambda2, q, gamma, termination_time,
                    tol=1e-4, max_iter=100, n_reps=20, eval_reps=None, verbose=False):
    """
    给定单个价格 p，返回：
      pi_m_star        : 该价格下不动点迭代收敛得到的稳态匹配概率
      avg_revenue_rate : eval_reps 次独立仿真得到的 revenue_rate 的样本均值
      std_revenue_rate : 同一组 revenue_rate 的样本标准差（衡量仿真噪声）

    每次 simulate(...) 内部会返回 revenue_rate = total_revenue / termination_time，
    即"这一次仿真在 [0, termination_time] 这段时间窗口里，平台每单位时间赚到的收益"。
    这里对 eval_reps 次独立重复仿真的 revenue_rate 取算术平均，
    用蒙特卡洛平均来抵消单次仿真中随机数（到达时刻、放弃与否、匹配成功与否）带来的噪声。
    """
    if eval_reps is None:
        eval_reps = n_reps

    pi_m_star, _ = fixed_point_solve(
        p, lambda1, lambda2, q, gamma, termination_time,
        tol=tol, max_iter=max_iter, n_reps=n_reps, verbose=verbose)

    revenue_rates = []
    for _ in range(eval_reps):
        _, _, _, _, _, revenue_rate = simulate(
            pi_m_star, p, lambda1, lambda2, q, gamma, termination_time)
        revenue_rates.append(revenue_rate)

    avg_revenue_rate = float(np.mean(revenue_rates))
    std_revenue_rate = float(np.std(revenue_rates))

    return pi_m_star, avg_revenue_rate, std_revenue_rate


# ---------------------------------------------------------------------
# 4. 扫描一系列价格 p，收集 revenue，并画图
# ---------------------------------------------------------------------
def sweep_prices_and_plot(p_values, lambda1, lambda2, q, gamma, termination_time,
                           n_reps=20, eval_reps=None, save_path="revenue_rate_vs_price.png"):
    """
    对 p_values 中的每一个价格点 p：
      1) 调用 evaluate_price(p, ...)：
         a. 先用 fixed_point_solve 在该价格下做不动点迭代，
            每次迭代内部跑 n_reps 次独立仿真，取平均放弃率，
            更新 pi_m，直到收敛（或达到 max_iter），得到该价格对应的
            稳态匹配概率 pi_m_star。
         b. 固定住这个 pi_m_star 后，再独立跑 eval_reps 次仿真
            （每次仿真都是从头开始的一次完整的 [0, termination_time]
            时间窗口模拟，事件到达、匹配、放弃的随机数各不相同）。
            每次仿真返回一个 revenue_rate = total_revenue / termination_time，
            即这一次仿真里，平台每单位时间的平均收益。
         c. 对这 eval_reps 个 revenue_rate 取样本均值 avg_revenue_rate
            和样本标准差 std_revenue_rate，作为该价格点最终的代表值
            和不确定性范围。
      2) 把 (p, avg_revenue_rate, std_revenue_rate) 存下来，用于画图。
    最终把所有价格点的 avg_revenue_rate 连成一条曲线画出来。
    """
    revenue_rates = []
    revenue_rate_stds = []
    pi_m_list = []

    for p in p_values:
        pi_m_star, avg_revenue_rate, std_revenue_rate = evaluate_price(
            p, lambda1, lambda2, q, gamma, termination_time,
            n_reps=n_reps, eval_reps=eval_reps, verbose=False)
        revenue_rates.append(avg_revenue_rate)
        revenue_rate_stds.append(std_revenue_rate)
        pi_m_list.append(pi_m_star)
        print(f"p={p:7.2f} -> pi_m*={pi_m_star:.4f}, "
              f"avg_revenue_rate={avg_revenue_rate:9.4f} (± {std_revenue_rate:.4f}, "
              f"基于 eval_reps={eval_reps if eval_reps else n_reps} 次独立仿真)")

    p_values = np.array(p_values)
    revenue_rates = np.array(revenue_rates)
    revenue_rate_stds = np.array(revenue_rate_stds)

    # 找到使收益率最大的价格点
    best_idx = int(np.argmax(revenue_rates))
    best_p = p_values[best_idx]
    best_rate = revenue_rates[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, revenue_rates, marker="o", color="#2563eb", linewidth=2,
            label="Average Revenue Rate")
    ax.fill_between(p_values, revenue_rates - revenue_rate_stds, revenue_rates + revenue_rate_stds,
                     color="#2563eb", alpha=0.15, label="±1 std")
    ax.scatter([best_p], [best_rate], color="#dc2626", zorder=5,
               label=f"Max at p={best_p:.1f}")

    ax.set_xlabel("Price p")
    ax.set_ylabel("Revenue Rate (revenue per unit time)")
    ax.set_title("Platform Revenue Rate vs. Price")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\n图像已保存至: {save_path}")

    return p_values, revenue_rates, revenue_rate_stds, pi_m_list


# ---------------------------------------------------------------------
# 5. 主程序入口
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 参数设置（与 revenue.py / goldenmethod.py 保持一致，可按需调整）
    lambda1 = 1
    lambda2 = 2
    q = 0.1
    gamma = 0.2
    termination_time = 1000
    n_reps = 20        # 用于不动点迭代内部的重复仿真次数
    eval_reps = 20     # 用于最终统计 revenue 的重复仿真次数

    # 价格扫描范围，可根据实际需要调整步长和范围
    p_values = np.arange(1, 15, 1)

    sweep_prices_and_plot(
        p_values, lambda1, lambda2, q, gamma, termination_time,
        n_reps=n_reps, eval_reps=eval_reps,
        save_path=r"C:\Users\s2829716\Documents\revenue_rate_vs_price.png"
    )
