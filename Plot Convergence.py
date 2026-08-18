import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fixpoint import fixed_point_solve

# Parameters
lambda1 = 1
lambda2 = 2
q = 0.1
gamma = 0.2
termination_time = 1000
p = 10
pi_m_init = 1.0
n_reps = 20
tol = 1e-4
max_iter = 100

pi_m_star, n_iter, history = fixed_point_solve(
    p, lambda1, lambda2, q, gamma, termination_time,
    tol=tol, max_iter=max_iter, n_reps=n_reps,
    pi_m_init=pi_m_init, return_history=True, verbose=True
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(len(history)), history, marker="o", markersize=4,
        linewidth=1.5, color="#2E75B6")
ax.axhline(pi_m_star, color="gray", linestyle="--", linewidth=1,
            label=f"Converged pi_m* = {pi_m_star:.4f}")
ax.set_xlabel("Iteration k")
ax.set_ylabel("pi_m")
ax.set_title(f"Convergence of pi_m (p={p}, lambda1={lambda1}, lambda2={lambda2}, "
             f"q={q}, gamma={gamma}, init={pi_m_init})")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("pi_m_convergence_custom.png", dpi=150)
print(f"\nSaved figure. Converged pi_m* = {pi_m_star:.4f} after {n_iter} iterations.")