"""
Two-sided pricing extension of the probabilistic matching system model
(see goldenmethod.py / fixedpoint1.py for the one-sided baseline).

Modeling changes vs. the one-sided baseline
--------------------------------------------
1. Employer (class-1) side: unchanged. Each employer has reservation value
   u1 ~ Uniform(0, U1_MAX). An employer joins iff u1 * pi_m >= p1, and is
   charged p1 upon joining.

2. Employee (class-2) side: NOW modeled symmetrically to the employer side
   instead of joining unconditionally. Each employee has reservation value
   u2 ~ Uniform(0, U2_MAX). An employee joins iff u2 * pi_m >= p2, and is
   charged p2 upon joining.

3. U2_MAX < U1_MAX (employees generally value the platform less than
   employers do), and the Nelder-Mead search box for p2 is bounded strictly
   below the search box for p1. This structurally guarantees that the
   price *range* offered to employees sits below the price range offered
   to employers, in addition to a pointwise penalty p2 <= p1 enforced
   inside the optimizer.

   IMPORTANT: U2_MAX is NOT taken from the project's baseline model (the
   baseline only specifies u1 ~ Uniform(0, 100) for employers; the
   employee-side reservation-value distribution is a new assumption
   introduced only for this two-sided extension). Since there is no
   data/theory in the project to pin its value down, U2_MAX is kept as an
   explicit, tunable parameter (default 30) rather than baked in, so a
   sensitivity sweep over it is a first-class experiment (see
   `sensitivity_over_u2_max` at the bottom of this file), not an
   afterthought.

4. pi_m (average platform-wide matching success probability) is still
   solved as a fixed point: given (p1, p2), simulate the system, update
   pi_m_new = 1 - abandoned/joined, iterate until convergence.

5. Instead of golden section search (only valid for 1-D unimodal search),
   we now have a 2-D decision variable (p1, p2), so we use the
   Nelder-Mead simplex method (derivative-free, works well with noisy
   black-box simulation objectives) to find the revenue-rate-maximizing
   price pair.
"""

import random
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# Reservation-value distributions (defaults; both are also passed
# explicitly as function arguments everywhere so they can be swept in
# sensitivity experiments without editing globals)
# ----------------------------------------------------------------------
U1_MAX_DEFAULT = 100.0   # employer reservation value ~ Uniform(0, U1_MAX)
U2_MAX_DEFAULT = 30.0    # employee reservation value ~ Uniform(0, U2_MAX)
                          # U2_MAX < U1_MAX  ->  employees are willing to pay
                          # less, which is what keeps the employee price
                          # range below the employer price range. This is an
                          # assumption, not a fitted/deriv python --versioned number -- treat
                          # it as a parameter to sweep, not a fixed truth.


def generate_exp(rate):
    return random.expovariate(rate)


# ----------------------------------------------------------------------
# 1. Simulation: both sides now have an admission (utility) test
# ----------------------------------------------------------------------
def simulate(pi_m, p1, p2, lambda1, lambda2, q, gamma, termination_time,
             u1_max=U1_MAX_DEFAULT, u2_max=U2_MAX_DEFAULT):
    """
    One simulation run of the two-sided-priced matching system.

    p1: price charged to employers (class-1)
    p2: price charged to employees (class-2)
    u1_max: employer reservation value ~ Uniform(0, u1_max)
    u2_max: employee reservation value ~ Uniform(0, u2_max) -- tunable,
            see module docstring for why this is not a fixed constant.

    Returns: total_joined, total_abandoned, Q1_end, Q2_end,
             total_revenue, revenue_rate
    """
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
    total_revenue = 0.0

    while t < termination_time:

        tn = min(event_calendar)
        col = event_calendar.index(tn)
        t = tn

        # ---------------- employer arrival ----------------
        if col == 0:
            u1 = random.uniform(0, u1_max)
            if u1 * pi_m >= p1:
                total_joined1 += 1
                total_revenue += p1
                uu = random.uniform(0, 1)

                if uu < (1 - q) ** Q2:          # fails to match -> waits
                    Q1 += 1
                    event_calendar.append(t + generate_exp(gamma))
                    direction.append(1)
                else:                            # matches immediately
                    Q2 -= 1
                    q2_indices = [i for i, d in enumerate(direction) if d == 2]
                    leave_idx = random.choice(q2_indices)
                    del event_calendar[leave_idx]
                    del direction[leave_idx]

            event_calendar[0] = t + generate_exp(lambda1)

        # ---------------- employee arrival ----------------
        elif col == 1:
            u2 = random.uniform(0, u2_max)
            if u2 * pi_m >= p2:
                total_joined2 += 1
                total_revenue += p2
                uu = random.uniform(0, 1)

                if uu < (1 - q) ** Q1:          # fails to match -> waits
                    Q2 += 1
                    event_calendar.append(t + generate_exp(gamma))
                    direction.append(2)
                else:                            # matches immediately
                    Q1 -= 1
                    q1_indices = [i for i, d in enumerate(direction) if d == 1]
                    leave_idx = random.choice(q1_indices)
                    del event_calendar[leave_idx]
                    del direction[leave_idx]

            event_calendar[1] = t + generate_exp(lambda2)

        # ---------------- termination ----------------
        elif col == 2:
            break

        # ---------------- abandonment ----------------
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


# ----------------------------------------------------------------------
# 2. Fixed point solve for pi_m given a price pair (p1, p2)
# ----------------------------------------------------------------------
def fixed_point_solve(p1, p2, lambda1, lambda2, q, gamma, termination_time,
                       tol=1e-4, max_iter=100, n_reps=20, verbose=False,
                       u1_max=U1_MAX_DEFAULT, u2_max=U2_MAX_DEFAULT):
    pi_m = 1.0

    for k in range(max_iter):
        total_joined = 0
        total_abandoned = 0

        for _ in range(n_reps):
            joined, abandoned, *_ = simulate(
                pi_m, p1, p2, lambda1, lambda2, q, gamma, termination_time,
                u1_max=u1_max, u2_max=u2_max)
            total_joined += joined
            total_abandoned += abandoned

        if total_joined == 0:
            if verbose:
                print(f"{k}: nobody joined in {n_reps} reps, stop iterating "
                      f"(p1={p1:.2f}, p2={p2:.2f})")
            return pi_m, k + 1

        pi_m_new = 1 - total_abandoned / total_joined
        diff = abs(pi_m_new - pi_m)

        if verbose:
            print(f"iter {k:3d}: pi_m {pi_m:.6f} -> {pi_m_new:.6f} "
                  f"(joined={total_joined}, abandoned={total_abandoned}, diff={diff:.6f})")

        pi_m = pi_m_new
        if diff < tol:
            if verbose:
                print(f"Converged at iteration {k}")
            break

    return pi_m, k + 1


# ----------------------------------------------------------------------
# 3. Evaluate a price pair: solve pi_m*, then estimate revenue rate at it
# ----------------------------------------------------------------------
def evaluate_prices(p1, p2, lambda1, lambda2, q, gamma, termination_time,
                     tol=1e-4, max_iter=100, n_reps=20, eval_reps=None,
                     crn_seeds=None, verbose=False,
                     u1_max=U1_MAX_DEFAULT, u2_max=U2_MAX_DEFAULT):
    if crn_seeds is None and eval_reps is None:
        eval_reps = n_reps

    pi_m_star, _ = fixed_point_solve(
        p1, p2, lambda1, lambda2, q, gamma, termination_time,
        tol=tol, max_iter=max_iter, n_reps=n_reps, verbose=verbose,
        u1_max=u1_max, u2_max=u2_max)

    total_revenue = 0.0
    total_revenue_rate = 0.0

    if crn_seeds is not None:
        # Common Random Numbers: same random streams for every (p1, p2)
        # candidate so the comparison across candidates is far less noisy,
        # which is important for Nelder-Mead since it has no gradient
        # information to smooth over simulation noise.
        for s in crn_seeds:
            random.seed(s)
            *_, revenue, revenue_rate = simulate(
                pi_m_star, p1, p2, lambda1, lambda2, q, gamma, termination_time,
                u1_max=u1_max, u2_max=u2_max)
            total_revenue += revenue
            total_revenue_rate += revenue_rate
        n_used = len(crn_seeds)
        random.seed()  # restore system randomness afterwards
    else:
        for _ in range(eval_reps):
            *_, revenue, revenue_rate = simulate(
                pi_m_star, p1, p2, lambda1, lambda2, q, gamma, termination_time,
                u1_max=u1_max, u2_max=u2_max)
            total_revenue += revenue
            total_revenue_rate += revenue_rate
        n_used = eval_reps

    avg_revenue = total_revenue / n_used
    avg_revenue_rate = total_revenue_rate / n_used
    return pi_m_star, avg_revenue, avg_revenue_rate


# ----------------------------------------------------------------------
# 4. Nelder-Mead search over (p1, p2) to maximize revenue rate
# ----------------------------------------------------------------------
def optimize_two_sided_prices(lambda1, lambda2, q, gamma, termination_time,
                               p1_bounds=(0.0, 100.0), p2_bounds=None,
                               u1_max=U1_MAX_DEFAULT, u2_max=U2_MAX_DEFAULT,
                               x0=None, n_reps=20, eval_reps=20,
                               use_crn=True, crn_reps=30, crn_seed=None,
                               maxiter=60, verbose=True):
    """
    Find (p1*, p2*) that maximizes the platform's revenue rate, using the
    Nelder-Mead simplex method (chosen because, once the decision variable
    grows from 1-D (golden section) to 2-D, and the objective is a noisy
    black-box simulation with no usable gradient, Nelder-Mead is the
    natural derivative-free extension).

    p2_bounds must sit at/under p1_bounds so that the employee price range
    stays below the employer price range by construction. If not given
    explicitly, it defaults to (0, u2_max) -- i.e. the search box for the
    employee price automatically follows whatever u2_max you pass in, so
    a sensitivity sweep over u2_max doesn't require also remembering to
    move p2_bounds by hand.
    """
    if p2_bounds is None:
        p2_bounds = (0.0, u2_max)

    assert p2_bounds[1] <= p1_bounds[1], (
        "employee price ceiling must not exceed the employer price ceiling")

    crn_seeds = None
    if use_crn:
        rng = random.Random(crn_seed)
        crn_seeds = [rng.randrange(1, 2**31 - 1) for _ in range(crn_reps)]

    cache = {}
    history = []

    def neg_revenue_rate(x):
        p1, p2 = x

        # Soft penalty as a safety net on top of scipy's own `bounds`
        # handling, and to additionally enforce p2 <= p1 pointwise (not
        # just range-wise), i.e. no employee should ever be charged more
        # than an employer under the same price pair.
        penalty = 0.0
        p1c = min(max(p1, p1_bounds[0]), p1_bounds[1])
        p2c = min(max(p2, p2_bounds[0]), p2_bounds[1])
        penalty += 1e4 * ((p1 - p1c) ** 2 + (p2 - p2c) ** 2)
        if p2c > p1c:
            penalty += 1e4 * (p2c - p1c)

        key = (round(p1c, 4), round(p2c, 4))
        if key in cache:
            rate = cache[key]
        else:
            _, _, rate = evaluate_prices(
                p1c, p2c, lambda1, lambda2, q, gamma, termination_time,
                n_reps=n_reps, eval_reps=eval_reps,
                crn_seeds=crn_seeds, verbose=False,
                u1_max=u1_max, u2_max=u2_max)
            cache[key] = rate

        history.append((p1c, p2c, rate))
        if verbose:
            print(f"  try p1={p1:7.3f} p2={p2:7.3f} -> revenue_rate={rate:.4f} "
                  f"(penalty={penalty:.3f})")
        return -rate + penalty

    if x0 is None:
        x0 = [
            0.5 * (p1_bounds[0] + p1_bounds[1]),
            0.5 * (p2_bounds[0] + p2_bounds[1]),
        ]

    result = minimize(
        neg_revenue_rate, x0=x0, method="Nelder-Mead",
        bounds=[p1_bounds, p2_bounds],
        options={"maxiter": maxiter, "xatol": 0.5, "fatol": 1e-3, "adaptive": True})

    p1_star = min(max(result.x[0], p1_bounds[0]), p1_bounds[1])
    p2_star = min(max(result.x[1], p2_bounds[0]), p2_bounds[1])

    # Final, higher-precision evaluation at the optimum
    pi_m_star, revenue_star, revenue_rate_star = evaluate_prices(
        p1_star, p2_star, lambda1, lambda2, q, gamma, termination_time,
        n_reps=n_reps, eval_reps=max(eval_reps, 40), crn_seeds=None, verbose=False,
        u1_max=u1_max, u2_max=u2_max)

    return p1_star, p2_star, pi_m_star, revenue_star, revenue_rate_star, result, history


# ----------------------------------------------------------------------
# 5. Sensitivity sweep over U2_MAX (the employee reservation-value ceiling)
# ----------------------------------------------------------------------
def sensitivity_over_u2_max(lambda1, lambda2, q, gamma, termination_time,
                             u2_max_values, u1_max=U1_MAX_DEFAULT,
                             p1_bounds=(0.0, 100.0),
                             n_reps=20, eval_reps=20,
                             use_crn=True, crn_reps=30, crn_seed=None,
                             maxiter=60, verbose=True):
    """
    Re-run the full pipeline (fixed point + Nelder-Mead) once per candidate
    u2_max value, so you can see how the optimal price pair (p1*, p2*),
    pi_m*, and the maximized revenue rate move as the employee-side
    reservation-value ceiling changes.

    Returns a list of dicts, one per u2_max value, each with keys:
    'u2_max', 'p1_star', 'p2_star', 'pi_m_star', 'revenue_star',
    'revenue_rate_star'.
    """
    results = []
    for u2_max in u2_max_values:
        if verbose:
            print(f"\n===== u2_max = {u2_max} =====")
        p1_star, p2_star, pi_m_star, revenue_star, revenue_rate_star, result, _ = \
            optimize_two_sided_prices(
                lambda1, lambda2, q, gamma, termination_time,
                p1_bounds=p1_bounds, p2_bounds=(0.0, u2_max),
                u1_max=u1_max, u2_max=u2_max,
                n_reps=n_reps, eval_reps=eval_reps,
                use_crn=use_crn, crn_reps=crn_reps, crn_seed=crn_seed,
                maxiter=maxiter, verbose=False)

        row = dict(u2_max=u2_max, p1_star=p1_star, p2_star=p2_star,
                   pi_m_star=pi_m_star, revenue_star=revenue_star,
                   revenue_rate_star=revenue_rate_star)
        results.append(row)

        if verbose:
            print(f"u2_max={u2_max:6.1f}  ->  p1*={p1_star:7.3f}  p2*={p2_star:7.3f}  "
                  f"pi_m*={pi_m_star:.4f}  revenue_rate*={revenue_rate_star:.4f}")

    return results


# ----------------------------------------------------------------------
# 6. When does charging employees (p2 > 0) start to beat keeping them
#    free (p2 = 0)?
# ----------------------------------------------------------------------
#
# Framing: this model characterizes long-run average (steady-state)
# behaviour for a FIXED set of primitives (lambda1, lambda2, q, gamma) --
# it is not a real-time controller reacting to the instantaneous queue.
# So "the state at which the platform should start charging employees"
# is best read as: the supply/demand REGIME the platform is operating in
# (summarized by rho = lambda2 / lambda1, the employee-to-employer
# arrival-rate ratio) crossing a threshold, beyond which employees are
# abundant enough relative to employers that extracting a small p2 > 0
# from them barely hurts matching quality but still adds revenue.
#
# We also report the resulting equilibrium pi_m and average queue levels
# at that regime, since those are the more directly observable "state"
# quantities a platform operator would actually watch.

def compare_charge_vs_free(lambda1, lambda2, q, gamma, termination_time,
                            u1_max=U1_MAX_DEFAULT, u2_max=U2_MAX_DEFAULT,
                            p1_bounds=(0.0, 100.0),
                            n_reps=20, eval_reps=20,
                            use_crn=True, crn_reps=30, crn_seed=None,
                            maxiter=40, improve_tol=1e-3, p2_eps=1e-2,
                            verbose=False):
    """
    For one system state (lambda1, lambda2, q, gamma), compare:

      Plan A ("free employees"): p2 fixed at 0, optimize p1 alone.
      Plan B ("two-sided"):      jointly optimize (p1, p2), p2 in [0, u2_max].

    Both plans use the SAME crn seed set, so the comparison is apples to
    apples (not just noise).

    Returns a dict with both optima, the improvement of B over A, and a
    boolean `charging_helps` flag (True iff B's optimal p2* is
    meaningfully positive AND B's revenue rate meaningfully beats A's).
    """
    crn_seeds = None
    if use_crn:
        rng = random.Random(crn_seed)
        crn_seeds = [rng.randrange(1, 2**31 - 1) for _ in range(crn_reps)]

    # ---- Plan A: p2 = 0, optimize p1 only (1-D Nelder-Mead) ----
    cache_a = {}

    def neg_rate_p1_only(x):
        p1c = min(max(x[0], p1_bounds[0]), p1_bounds[1])
        key = round(p1c, 4)
        if key not in cache_a:
            _, _, rate = evaluate_prices(
                p1c, 0.0, lambda1, lambda2, q, gamma, termination_time,
                n_reps=n_reps, eval_reps=eval_reps, crn_seeds=crn_seeds,
                verbose=False, u1_max=u1_max, u2_max=u2_max)
            cache_a[key] = rate
        return -cache_a[key]

    result_a = minimize(
        neg_rate_p1_only, x0=[0.5 * (p1_bounds[0] + p1_bounds[1])],
        method="Nelder-Mead", bounds=[p1_bounds],
        options={"maxiter": maxiter, "xatol": 0.5, "fatol": 1e-3, "adaptive": True})
    p1_a = min(max(result_a.x[0], p1_bounds[0]), p1_bounds[1])
    pi_m_a, revenue_a, rate_a = evaluate_prices(
        p1_a, 0.0, lambda1, lambda2, q, gamma, termination_time,
        n_reps=n_reps, eval_reps=max(eval_reps, 40), crn_seeds=None,
        u1_max=u1_max, u2_max=u2_max)

    # ---- Plan B: jointly optimize (p1, p2) ----
    p1_b, p2_b, pi_m_b, revenue_b, rate_b, result_b, _ = optimize_two_sided_prices(
        lambda1, lambda2, q, gamma, termination_time,
        p1_bounds=p1_bounds, p2_bounds=(0.0, u2_max),
        u1_max=u1_max, u2_max=u2_max,
        n_reps=n_reps, eval_reps=eval_reps,
        use_crn=use_crn, crn_reps=crn_reps, crn_seed=crn_seed,
        maxiter=maxiter, verbose=False)

    improvement = rate_b - rate_a
    charging_helps = (p2_b > p2_eps) and (improvement > improve_tol)

    row = dict(
        lambda1=lambda1, lambda2=lambda2, rho=lambda2 / lambda1,
        p1_free=p1_a, pi_m_free=pi_m_a, rate_free=rate_a,
        p1_charge=p1_b, p2_charge=p2_b, pi_m_charge=pi_m_b, rate_charge=rate_b,
        improvement=improvement, charging_helps=charging_helps,
    )

    if verbose:
        print(f"rho={row['rho']:.2f}: free-> p1={p1_a:.2f} rate={rate_a:.4f} | "
              f"charge-> p1={p1_b:.2f} p2={p2_b:.2f} rate={rate_b:.4f} | "
              f"gain={improvement:+.4f} | charging_helps={charging_helps}")

    return row


def sweep_state_grid(lambda1, rho_values, q, gamma, termination_time,
                      u1_max=U1_MAX_DEFAULT, u2_max=U2_MAX_DEFAULT,
                      **kwargs):
    """
    Run compare_charge_vs_free across a grid of rho = lambda2/lambda1
    values (lambda1 held fixed, lambda2 = rho * lambda1). Use this FIRST
    to sanity-check that 'charging_helps' is monotone in rho before
    trusting the bisection search below -- if it isn't monotone, there
    isn't a single clean threshold and the bisection result is not
    meaningful.
    """
    results = []
    for rho in rho_values:
        lambda2 = rho * lambda1
        row = compare_charge_vs_free(
            lambda1, lambda2, q, gamma, termination_time,
            u1_max=u1_max, u2_max=u2_max, verbose=True, **kwargs)
        results.append(row)
    return results


def find_charging_threshold(lambda1, rho_low, rho_high, q, gamma, termination_time,
                             u1_max=U1_MAX_DEFAULT, u2_max=U2_MAX_DEFAULT,
                             tol=0.05, max_iter=15, verbose=True, **kwargs):
    """
    Bisection search over rho = lambda2 / lambda1 (employee-to-employer
    arrival-rate ratio) for the threshold state at which charging
    employees starts to be the better plan.

    Assumes charging_helps(rho) is False for small rho and True for large
    rho (employees relatively scarce -> keep them free to protect
    liquidity; employees relatively abundant -> can afford to charge
    them). This monotonicity should be checked with sweep_state_grid
    first; if lo and hi give the SAME charging_helps value, there is no
    single crossover in [rho_low, rho_high] and this function returns
    None along with the two boundary evaluations for you to inspect.

    Returns: (rho_threshold_or_None, history)
    history is the list of every state evaluated (dicts from
    compare_charge_vs_free), in the order evaluated.
    """
    def helps_at(rho):
        lambda2 = rho * lambda1
        row = compare_charge_vs_free(
            lambda1, lambda2, q, gamma, termination_time,
            u1_max=u1_max, u2_max=u2_max, verbose=verbose, **kwargs)
        return row["charging_helps"], row

    lo, hi = rho_low, rho_high
    lo_helps, lo_row = helps_at(lo)
    hi_helps, hi_row = helps_at(hi)
    history = [lo_row, hi_row]

    if lo_helps == hi_helps:
        if verbose:
            print(f"No crossover detected in rho in [{rho_low}, {rho_high}]: "
                  f"charging_helps is {lo_helps} at both ends. "
                  f"Widen the search range or check monotonicity with sweep_state_grid.")
        return None, history

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mid_helps, mid_row = helps_at(mid)
        history.append(mid_row)

        if mid_helps == lo_helps:
            lo = mid
        else:
            hi = mid

        if hi - lo < tol:
            break

    rho_threshold = 0.5 * (lo + hi)
    if verbose:
        print(f"\nThreshold located at rho* = lambda2/lambda1 ~= {rho_threshold:.4f} "
              f"(interval width {hi - lo:.4f})")
    return rho_threshold, history


# ----------------------------------------------------------------------
if __name__ == "__main__":
    lambda1 = 1
    lambda2 = 2
    q = 0.1
    gamma = 0.2
    termination_time = 1000
    n_reps = 20

    print("=== Sanity check: fixed point at p1=20, p2=5 ===")
    pi_m_star, n_iter = fixed_point_solve(
        20, 5, lambda1, lambda2, q, gamma, termination_time,
        tol=1e-4, max_iter=100, n_reps=n_reps, verbose=True)
    print(f"pi_m* = {pi_m_star:.4f} ({n_iter} iterations)\n")

    print("=== Nelder-Mead search for (p1*, p2*) maximizing revenue rate ===")
    p1_star, p2_star, pi_m_at_star, revenue_star, revenue_rate_star, result, history = \
        optimize_two_sided_prices(
            lambda1, lambda2, q, gamma, termination_time,
            p1_bounds=(0.0, 100.0), p2_bounds=(0.0, 30.0),
            n_reps=n_reps, eval_reps=30,
            use_crn=True, crn_reps=30,
            maxiter=60, verbose=True)

    print(f"\nOptimal price pair: p1* = {p1_star:.3f} (employer), "
          f"p2* = {p2_star:.3f} (employee)")
    print(f"pi_m*(p1*, p2*)   = {pi_m_at_star:.4f}")
    print(f"avg_revenue       = {revenue_star:.2f}")
    print(f"avg_revenue_rate  = {revenue_rate_star:.4f}")
    print(f"Nelder-Mead status: {result.message} (nit={result.nit}, nfev={result.nfev})")

    # ---- Step 3: sensitivity sweep over U2_MAX (employee reservation
    #      value ceiling) -- U2_MAX is an assumption, not a fitted number,
    #      so instead of hard-coding it we sweep it and see how the
    #      optimal price pair and revenue rate respond. ----
    print("\n=== Sensitivity sweep over U2_MAX ===")
    u2_max_grid = [10, 20, 30, 50, 70, 100]  # 100 = symmetric with employer
    sweep_results = sensitivity_over_u2_max(
        lambda1, lambda2, q, gamma, termination_time,
        u2_max_values=u2_max_grid, u1_max=100.0,
        n_reps=n_reps, eval_reps=30,
        use_crn=True, crn_reps=30, maxiter=60, verbose=True)

    print("\nu2_max |    p1*    p2*  | pi_m*  | revenue_rate*")
    print("-------+-----------------+--------+--------------")
    for row in sweep_results:
        print(f"{row['u2_max']:6.1f} | {row['p1_star']:6.2f} {row['p2_star']:6.2f} "
              f"| {row['pi_m_star']:.4f} | {row['revenue_rate_star']:.4f}")

    # ---- Step 4: when does charging employees start to beat keeping
    #      them free? First check monotonicity in rho = lambda2/lambda1
    #      with a coarse grid, then bisect to pin down the threshold. ----
    print("\n=== Step 4a: grid check of charging_helps(rho) for monotonicity ===")
    rho_grid = [1.0, 2.0, 4.0, 8.0, 16.0]
    grid_results = sweep_state_grid(
        lambda1=1, rho_values=rho_grid, q=q, gamma=gamma,
        termination_time=termination_time,
        n_reps=n_reps, eval_reps=30, use_crn=True, crn_reps=30, maxiter=40)

    print("\n=== Step 4b: bisection search for the rho threshold ===")
    rho_star, threshold_history = find_charging_threshold(
        lambda1=1, rho_low=1.0, rho_high=16.0, q=q, gamma=gamma,
        termination_time=termination_time,
        n_reps=n_reps, eval_reps=30, use_crn=True, crn_reps=30,
        maxiter=40, tol=0.1, max_iter=12, verbose=True)

    if rho_star is not None:
        print(f"\n>>> Below rho ~= {rho_star:.2f} (employees not abundant enough "
              f"relative to employers), keep p2 = 0.")
        print(f">>> Above rho ~= {rho_star:.2f}, switch to p2 > 0.")
        # Report the equilibrium pi_m / congestion right around the threshold
        # as the more directly observable "system state" signal.
        near = min(threshold_history, key=lambda r: abs(r["rho"] - rho_star))
        print(f">>> At that regime, equilibrium pi_m ~= {near['pi_m_free']:.4f} "
              f"(under the free-employee plan just before switching).")
    else:
        print("\nNo crossover found in the searched rho range -- "
              "widen rho_low/rho_high and retry.")