import matplotlib.pyplot as plt

# 检验原始固定点法（goldenmethod.py 里只有一个 pi_m 的版本）本身
# 是否存在和加权模型一样的采样噪声/震荡现象。
# simulate() 直接复用仓库原版，不做任何修改；只是把 fixed_point_solve()
# 的每一轮迭代过程记录下来（原版函数不返回 history，这里加一层记录），
# 迭代逻辑和收敛判据与原版完全一致，没有改动任何计算方式。
from goldenmethod import simulate


def fixed_point_solve_diag(p, lambda1, lambda2, q, gamma, termination_time,
                            tol=1e-2, max_iter=100, n_reps=20):
    pi_m = 1.0
    history = [(0, pi_m, None)]  # 第0轮：初始值，还没有 diff

    for k in range(1, max_iter + 1):
        total_joined = 0
        total_abandoned = 0

        for _ in range(n_reps):
            joined, abandoned, Q1_end, Q2_end, revenue, revenue_rate = simulate(
                pi_m, p, lambda1, lambda2, q, gamma, termination_time)
            total_joined += joined
            total_abandoned += abandoned

        if total_joined == 0:
            print(f"  [p={p}] 第{k}轮: 无人加入，提前终止")
            break

        pi_m_new = 1 - total_abandoned / total_joined
        diff = abs(pi_m_new - pi_m)
        pi_m = pi_m_new
        history.append((k, pi_m, diff))

        if diff < tol:
            print(f"  [p={p}] {k} {pi_m} (diff={diff:.6f})")
        if k==max_iter:
            break
    
    return pi_m, history


# ---------------------------------------------------------------------
# 参数与 check_convergence_pi_m1_pi_m.py 保持一致，方便直接对比
# ---------------------------------------------------------------------
q = 0.1
gamma = 0.2
termination_time = 1000
n_reps = 20
p_test = 15

test_cases = [
    
    ("Convergence Check", 1, 2),
    
]

plt.figure(figsize=(12, 8))

for idx, (label, lambda1, lambda2) in enumerate(test_cases):
    print(f"\n=== {label}: lambda1={lambda1}, lambda2={lambda2} ===")
    pi_m_final, history = fixed_point_solve_diag(
        p_test, lambda1, lambda2, q, gamma, termination_time,
        tol=1e-2, max_iter=100, n_reps=n_reps)

    iters = [h[0] for h in history]
    pi_m_hist = [h[1] for h in history]
    diff_hist = [h[2] for h in history[1:]]  # 跳过第0轮（没有diff）

    # 子图1：pi_m 的轨迹
    plt.subplot(2, 3, idx + 1)
    plt.plot(iters, pi_m_hist, marker='o', markersize=3, color='tab:blue')
    plt.xlabel("iteration")
    plt.ylabel("pi_m")
    plt.title(f"{label}\ntrajectory")
    plt.grid(True)

    # 子图2：diff 随迭代次数变化（log坐标）
    plt.subplot(2, 3, idx + 4)
    plt.plot(iters[1:], diff_hist, marker='o', markersize=3, color='tab:blue')
    plt.axhline(y=1e-2, color='gray', linestyle='--', linewidth=1, label='tol=1e-2')
    plt.yscale('log')
    plt.xlabel("iteration")
    plt.ylabel("|diff| (log scale)")
    plt.title("convergence diff")
    plt.legend(fontsize=8)
    plt.grid(True, which='both')

plt.tight_layout()
plt.savefig("pi_m_original_convergence_check.png", dpi=150)
plt.show()