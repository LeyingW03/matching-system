import numpy as np
import matplotlib.pyplot as plt
from goldenmethod import fixed_point_solve   # 直接用不动点法，不做价格搜索

lambda1, lambda2, gamma = 1, 4, 0.2
termination_time = 1000
p_fixed = 15   # 固定一个中等价格，不参与优化，避免价格搜索噪声混进来
n_reps = 50

q_values = np.arange(0.1, 1.0, 0.1)
pi_m_list = []
for q in q_values:
    pi_m_star, _ = fixed_point_solve(p_fixed, lambda1, lambda2, q, gamma,
                                      termination_time, n_reps=n_reps)
    pi_m_list.append(pi_m_star)

plt.plot(q_values, pi_m_list, marker='o')
plt.axvline(lambda1/lambda2, color='red', linestyle='--', label=f'q=λ1/λ2={lambda1/lambda2}')
plt.xlabel("q"); plt.ylabel("pi_m*"); plt.legend(); plt.grid(True)
plt.savefig("pi_m_vs_q_fixed_price.png", dpi=150)
plt.show()