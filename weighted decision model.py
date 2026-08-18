import random
import numpy as np
import matplotlib.pyplot as plt

# 本脚本实现"方向2"：雇主的加入决策不再只看整体 pi_m，而是同时看
# 雇主端自己的匹配率 pi_m1（权重 w，w 较大）和整体匹配率 pi_m（权重 1-w）：
#
#     u * (w * pi_m1 + (1-w) * pi_m) > p
#
# 因为决策规则本身变了，simulate() / fixed_point_solve() / golden_section_search()
# 都要相应修改，不能直接复用 goldenmethod.py 里的原版（原版只有一个 pi_m）。
# 下面是这三个函数的"加权版"实现，结构和原版保持一致，方便对照。
#
# 只有 golden_section_search 的整体框架（黄金分割搜索本身）沿用了
# goldenmethod.py 里的算法逻辑，这里重新实现是因为它内部调用的
# evaluate_price 需要换成加权版的 fixed point。

from goldenmethod import golden_section_search as golden_section_search_orig


def generate_exp(rate):
    return random.expovariate(rate)


# ---------------------------------------------------------------------
# 1. simulate_v2: 加权决策版的仿真
#    唯一改动：雇主加入判断从 u*pi_m>p 改为 u*(w*pi_m1+(1-w)*pi_m)>p
#    其余匹配/放弃机制与原版完全一致
# ---------------------------------------------------------------------
def simulate_v2(pi_m1, pi_m, p, lambda1, lambda2, q, gamma, w, termination_time):
    Q1, Q2 = 0, 0
    event_calendar = [generate_exp(lambda1), generate_exp(lambda2), termination_time]
    direction = [0, 0, 0]
    t = 0.0

    joined1, joined2 = 0, 0
    abandoned1, abandoned2 = 0, 0
    total_revenue = 0.0

    pi_eff = w * pi_m1 + (1 - w) * pi_m  # 雇主决策时实际使用的加权匹配率

    while t < termination_time:
        tn = min(event_calendar)
        col = event_calendar.index(tn)
        t = tn

        if col == 0:  # 雇主到达
            u = random.uniform(0, 100)
            if u * pi_eff > p:
                joined1 += 1
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

        elif col == 1:  # 雇员到达（不受价格/匹配率影响，无条件加入）
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

        else:  # 放弃
            if direction[col] == 1:
                Q1 -= 1
                abandoned1 += 1
            else:
                Q2 -= 1
                abandoned2 += 1
            del event_calendar[col]
            del direction[col]

    revenue_rate = total_revenue / termination_time
    return joined1, joined2, abandoned1, abandoned2, total_revenue, revenue_rate


# ---------------------------------------------------------------------
# 2. fixed_point_solve_v2: (pi_m1, pi_m) 联合不动点
#    单层循环，两个量在同一轮里用同一批仿真结果一起更新（不是嵌套循环）
# ---------------------------------------------------------------------
def fixed_point_solve_v2(p, lambda1, lambda2, q, gamma, w, termination_time,
                          tol=1e-4, max_iter=100, n_reps=20, damping=1.0):
    pi_m1, pi_m = 1.0, 1.0

    for k in range(max_iter):
        j1_sum, j2_sum, a1_sum, a2_sum = 0, 0, 0, 0
        for _ in range(n_reps):
            j1, j2, a1, a2, _, _ = simulate_v2(
                pi_m1, pi_m, p, lambda1, lambda2, q, gamma, w, termination_time)
            j1_sum += j1
            j2_sum += j2
            a1_sum += a1
            a2_sum += a2

        if j1_sum == 0:
            # 价格太高，雇主没人加入：无法更新 pi_m1，直接停止
            return pi_m1, pi_m, k + 1

        total_joined = j1_sum + j2_sum
        total_abandoned = a1_sum + a2_sum

        pi_m1_new = 1 - a1_sum / j1_sum
        pi_m_new = 1 - total_abandoned / total_joined if total_joined > 0 else pi_m

        diff1 = abs(pi_m1_new - pi_m1)
        diff = abs(pi_m_new - pi_m)

        # damping=1.0 即不加阻尼；如遇震荡不收敛可调小（如0.5）
        pi_m1 = pi_m1 + damping * (pi_m1_new - pi_m1)
        pi_m = pi_m + damping * (pi_m_new - pi_m)

        if diff1 < tol and diff < tol:
            break

    return pi_m1, pi_m, k + 1


# ---------------------------------------------------------------------
# 3. evaluate_price_v2: 给定价格 p，求 (pi_m1*, pi_m*)，再评估平均收益率
#    （沿用原版 evaluate_price 的 CRN 思路，让黄金分割搜索的价格比较更稳定）
# ---------------------------------------------------------------------
def evaluate_price_v2(p, lambda1, lambda2, q, gamma, w, termination_time,
                       tol=1e-4, max_iter=100, n_reps=20, crn_seeds=None, eval_reps=None):
    pi_m1_star, pi_m_star, _ = fixed_point_solve_v2(
        p, lambda1, lambda2, q, gamma, w, termination_time,
        tol=tol, max_iter=max_iter, n_reps=n_reps)

    total_revenue, total_revenue_rate = 0.0, 0.0
    if crn_seeds is not None:
        for s in crn_seeds:
            random.seed(s)
            _, _, _, _, revenue, revenue_rate = simulate_v2(
                pi_m1_star, pi_m_star, p, lambda1, lambda2, q, gamma, w, termination_time)
            total_revenue += revenue
            total_revenue_rate += revenue_rate
        n_used = len(crn_seeds)
        random.seed()
    else:
        n_used = eval_reps if eval_reps is not None else n_reps
        for _ in range(n_used):
            _, _, _, _, revenue, revenue_rate = simulate_v2(
                pi_m1_star, pi_m_star, p, lambda1, lambda2, q, gamma, w, termination_time)
            total_revenue += revenue
            total_revenue_rate += revenue_rate

    avg_revenue_rate = total_revenue_rate / n_used
    return pi_m1_star, pi_m_star, avg_revenue_rate


# ---------------------------------------------------------------------
# 4. golden_section_search_v2: 黄金分割搜索最优价格 p*（算法逻辑与
#    goldenmethod.py 中的 golden_section_search 相同，只是内部换成 v2 的
#    evaluate_price_v2）
# ---------------------------------------------------------------------
def golden_section_search_v2(lambda1, lambda2, q, gamma, w, termination_time,
                              p_low, p_high, tol=1.0, max_iter=50,
                              n_reps=20, eval_reps=None,
                              use_crn=True, crn_reps=30, crn_seed=None):
    gr = (5 ** 0.5 - 1) / 2
    a, b = p_low, p_high

    crn_seeds = None
    if use_crn:
        rng = random.Random(crn_seed)
        crn_seeds = [rng.randrange(1, 2**31 - 1) for _ in range(crn_reps)]

    cache = {}

    def f(p):
        key = round(p, 6)
        if key in cache:
            return cache[key]
        pi_m1_star, pi_m_star, avg_revenue_rate = evaluate_price_v2(
            p, lambda1, lambda2, q, gamma, w, termination_time,
            n_reps=n_reps, eval_reps=eval_reps, crn_seeds=crn_seeds)
        cache[key] = (avg_revenue_rate, pi_m1_star, pi_m_star)
        return cache[key]

    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)[0]
    fd = f(d)[0]

    for _ in range(max_iter):
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)[0]
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)[0]
        if abs(b - a) < tol:
            break

    p_star = (a + b) / 2
    revenue_rate_star, pi_m1_star, pi_m_star = f(p_star)
    return p_star, pi_m1_star, pi_m_star, revenue_rate_star


# ---------------------------------------------------------------------
# base case 参数
# ---------------------------------------------------------------------
q = 0.1
gamma = 0.2
termination_time = 1000
n_reps = 20
eval_reps = 20
p_low, p_high = 0, 100
tol = 1
w = 0.8   # 雇主决策对"自己端匹配率 pi_m1"的权重，(1-w)=0.2 给整体 pi_m

lambda1 = 1
ratio_values = [0.5, 0.75, 1, 1.5, 2, 3, 5, 8]
lambda2_values = [lambda1 * r for r in ratio_values]

# ---------------------------------------------------------------------
# 对每个 ratio：分别跑"原模型（只用整体 pi_m）"和"新模型（加权决策）"，
# 方便在图上直接对比两者的差异
# ---------------------------------------------------------------------
p_star_orig, revenue_rate_orig = [], []
p_star_v2, revenue_rate_v2, pi_m1_v2, pi_m_v2 = [], [], [], []

for ratio, lambda2 in zip(ratio_values, lambda2_values):
    # 原模型（仓库自带，未改动）
    p_o, _, _, rr_o, _ = golden_section_search_orig(
        lambda1, lambda2, q, gamma, termination_time,
        p_low=p_low, p_high=p_high, tol=tol,
        n_reps=n_reps, eval_reps=eval_reps, verbose=False)
    p_star_orig.append(p_o)
    revenue_rate_orig.append(rr_o)

    # 新模型（加权决策 + 联合不动点）
    p_v, pm1_v, pm_v, rr_v = golden_section_search_v2(
        lambda1, lambda2, q, gamma, w, termination_time,
        p_low=p_low, p_high=p_high, tol=tol,
        n_reps=n_reps, eval_reps=eval_reps)
    p_star_v2.append(p_v)
    revenue_rate_v2.append(rr_v)
    pi_m1_v2.append(pm1_v)
    pi_m_v2.append(pm_v)

    print(f"ratio={ratio:>4.2f} | orig: p*={p_o:6.3f}, R*={rr_o:6.3f}  ||  "
          f"v2(w={w}): p*={p_v:6.3f}, R*={rr_v:6.3f}, "
          f"pi_m1*={pm1_v:.4f}, pi_m*={pm_v:.4f}")

# ---------------------------------------------------------------------
# 画图 1：p* 对比（原模型 vs 加权决策新模型）
# ---------------------------------------------------------------------
plt.figure(figsize=(7.5, 5.5))
plt.plot(ratio_values, p_star_v2, marker='s', label=f'weighted model (w={w} on pi_m1)')
plt.axvline(x=2, color='gray', linestyle=':', linewidth=1, label='base case (ratio=2)')
plt.xlabel("ratio = lambda2 / lambda1")
plt.ylabel("optimal price p*")
plt.title("Optimal price p*: original vs weighted decision rule\n"
          f"(lambda1={lambda1}, q={q}, gamma={gamma})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pstar_orig_vs_weighted.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 画图 2：revenue_rate* 对比
# ---------------------------------------------------------------------
plt.figure(figsize=(7.5, 5.5))
plt.plot(ratio_values, revenue_rate_v2, marker='s', color='tab:orange',
         label=f'weighted model (w={w})')
plt.axvline(x=2, color='gray', linestyle=':', linewidth=1, label='base case (ratio=2)')
plt.xlabel("ratio = lambda2 / lambda1")
plt.ylabel("optimal revenue rate")
plt.title("Optimal revenue rate: original vs weighted decision rule\n"
          f"(lambda1={lambda1}, q={q}, gamma={gamma})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("revenue_rate_orig_vs_weighted.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 画图 3：新模型下 pi_m1* 与 pi_m* 分别随 ratio 的变化，
# 直接展示"雇主端体验"与"整体口碑"之间的差距如何随 ratio 拉开
# ---------------------------------------------------------------------
plt.figure(figsize=(7.5, 5.5))
plt.plot(ratio_values, pi_m1_v2, marker='o', label='pi_m1* (employer-side)')
plt.plot(ratio_values, pi_m_v2, marker='s', label='pi_m* (platform-wide)')
plt.axvline(x=2, color='gray', linestyle=':', linewidth=1, label='base case (ratio=2)')
plt.xlabel("ratio = lambda2 / lambda1")
plt.ylabel("matching probability")
plt.title(f"pi_m1* vs pi_m* under weighted model (w={w})\n"
          f"(lambda1={lambda1}, q={q}, gamma={gamma})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pi_m1_vs_pi_m_weighted.png", dpi=150)
plt.show()