import random

# Generate random numbers for the arrival rate
def generate_exp(rate):
    return random.expovariate(rate)

# 1: simulation process
def simulate(pi_m, p, lambda1, lambda2, q, gamma, termination_time):
    Q1 = 0
    Q2 = 0

    event_calendar = [
        generate_exp(lambda1),
        generate_exp(lambda2),
        termination_time,]
    
    direction = [0, 0, 0]  

    t = 0.0
    total_joined1 = 0       # number of employers actually join (after utility judgment)
    total_joined2 = 0       # number of employees actually join (no utility judgment)
    total_abandoned = 0
    total_revenue = 0.0     # platform's total revenue, only employers are charged (one-sided pricing)

    while t < termination_time:

        tn = min(event_calendar)
        col = event_calendar.index(tn)
        t = tn

        # employer side
        if col == 0:
            # First, utility judgment, determine joining the system or not
            u = random.uniform(0, 100)  # utility of this employer
            if u * pi_m > p:
                total_joined1 += 1
                total_revenue += p  # once an employer decides to join, platform revenue increases by p
                uu = random.uniform(0, 1) # Generate exclusive matching value

                # Second, after joining, determine matching or not
                if uu < (1 - q) ** Q2: # failed
                    Q1 += 1
                    event_calendar.append(t + generate_exp(gamma))
                    direction.append(1)

                else: # success
                    Q2 -= 1
                    q2_indices = [i for i, d in enumerate(direction) if d == 2]
                    leave_idx = random.choice(q2_indices)
                    del event_calendar[leave_idx]
                    del direction[leave_idx]

            # Whether joining or not, arrange the next event time:
            event_calendar[0] = t + generate_exp(lambda1)

        # employee side
        elif col == 1:
            # No utility judgment, join the system directly
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

        # Termination
        elif col == 2:
            break

        # Abandon moment
        else:
            total_abandoned += 1
            if direction[col] == 1:
                Q1 -= 1
            else:
                Q2 -= 1
            del event_calendar[col]
            del direction[col]

    total_joined = total_joined1 + total_joined2
    revenue_rate = total_revenue / termination_time  # revenue per unit time of this single run
    return total_joined, total_abandoned, Q1, Q2, total_revenue, revenue_rate

# 2: Fixpoint calculation
def fixed_point_solve(p, lambda1, lambda2, q, gamma, termination_time,
                       tol=1e-4, max_iter=100, n_reps=20, verbose=True):
    
    # initial pi_m value
    pi_m = 1.0

    # Perform max_iter iterations
    for k in range(max_iter):
        total_joined = 0
        total_abandoned = 0
        total_revenue = 0.0       # sum of platform revenue over n_reps simulations
        total_revenue_rate = 0.0  # sum of revenue rate over n_reps simulations

        # Perform n_rep independent simulations for each iteration
        for _ in range(n_reps):
            joined, abandoned, Q1_end, Q2_end, revenue, revenue_rate = simulate(
                pi_m, p, lambda1, lambda2, q, gamma, termination_time)
            # Calculate the total_joined and total_abandonedsum of n_rep simulations
            total_joined += joined
            total_abandoned += abandoned
            total_revenue += revenue
            total_revenue_rate += revenue_rate

        # Extreme situation: Price too high, no one joining
        if total_joined == 0:
            if verbose:
                print(f"{k}: No one joined in {n_reps}th repeated simulation, iteration stopped")

            # p is too high, nobody joins -> revenue is 0, no meaningful pi_m to update.
            # Return pi_m unchanged (or 1.0) together with revenue = 0 so the caller
            # (golden section search) can still treat this price consistently.
            return pi_m, k + 1

        # Calculate new pi_m
        pi_m_new = 1 - total_abandoned / total_joined
        diff = abs(pi_m_new - pi_m)

        # Average platform revenue and revenue rate of this iteration (over n_reps simulations)
        avg_revenue = total_revenue / n_reps
        avg_revenue_rate = total_revenue_rate / n_reps

        # Print calculation details
        if verbose:
            print(f"iter {k:3d}: {pi_m_new:.6f} "
                  f"(Totaljoined={total_joined}, abandoned={total_abandoned}, "
                  f"diff={diff:.6f}, "
                  f"revenue={avg_revenue:.2f}, revenue_rate={avg_revenue_rate:.4f})")
            
        # Update pi_m in the Kth iteration
        pi_m = pi_m_new
        
        # Determine whether the convergence requirements have been met
        if diff < tol:
            if verbose:
                print(f"Converge to the {k}th iteration")
            break

    return pi_m, k + 1


# 3: Given a price p, solve for the fixed point pi_m* and report the
#    average steady-state revenue rate under that price.
#    This is the "f(p)" that golden section search will call repeatedly.
def evaluate_price(p, lambda1, lambda2, q, gamma, termination_time,
                    tol=1e-4, max_iter=100, n_reps=20, eval_reps=None, verbose=False,
                    crn_seeds=None):
    """
    Solve the fixed point pi_m*(p), then run additional simulations at
    pi_m*(p) to estimate the average revenue and average revenue rate
    for this price p.

    eval_reps: number of extra simulations used to *evaluate* revenue at
               the converged pi_m*. Defaults to n_reps if not given.
    crn_seeds: optional list of random seeds (Common Random Numbers). If
               given, len(crn_seeds) simulations are run, one per seed,
               with random.seed(seed) set immediately before each simulate()
               call. Using the SAME seed list across different prices p
               means every price is "tested against the same luck", which
               greatly reduces the noise in comparing f(p1) vs f(p2) -
               this is what makes golden section search stable.
               If given, it overrides eval_reps (len(crn_seeds) is used).
    Returns: (pi_m_star, avg_revenue, avg_revenue_rate)
    """
    if crn_seeds is None and eval_reps is None:
        eval_reps = n_reps

    pi_m_star, _ = fixed_point_solve(
        p, lambda1, lambda2, q, gamma, termination_time,
        tol=tol, max_iter=max_iter, n_reps=n_reps, verbose=verbose)

    total_revenue = 0.0
    total_revenue_rate = 0.0

    if crn_seeds is not None:
        # Common Random Numbers: replay the exact same random streams for
        # every price so differences reflect p, not luck.
        for s in crn_seeds:
            random.seed(s)
            _, _, _, _, revenue, revenue_rate = simulate(
                pi_m_star, p, lambda1, lambda2, q, gamma, termination_time)
            total_revenue += revenue
            total_revenue_rate += revenue_rate
        n_used = len(crn_seeds)
        # re-seed with system randomness afterwards so later unrelated
        # code (e.g. other prices, other runs) isn't accidentally frozen
        random.seed()
    else:
        for _ in range(eval_reps):
            _, _, _, _, revenue, revenue_rate = simulate(
                pi_m_star, p, lambda1, lambda2, q, gamma, termination_time)
            total_revenue += revenue
            total_revenue_rate += revenue_rate
        n_used = eval_reps

    avg_revenue = total_revenue / n_used
    avg_revenue_rate = total_revenue_rate / n_used

    return pi_m_star, avg_revenue, avg_revenue_rate


# 4: Golden section search for the price p that maximizes the average
#    revenue rate.
def golden_section_search(lambda1, lambda2, q, gamma, termination_time,
                           p_low, p_high, tol=1e-2, max_iter=50,
                           n_reps=20, eval_reps=None, verbose=True,
                           use_crn=True, crn_reps=30, crn_seed=None):
    """
    Search for p* in [p_low, p_high] that maximizes avg_revenue_rate,
    using golden section search. Assumes the revenue-rate curve is
    (approximately) unimodal in p.

    use_crn: if True (default), generate one fixed list of `crn_reps`
             random seeds ONCE for this whole search call, and reuse that
             SAME seed list to evaluate every price p that gets tried.
             This is the Common Random Numbers variance-reduction trick:
             every price is simulated under the "same luck", so the
             comparison between prices is much less noisy and the search
             converges to a more repeatable p* across runs.
             If False, falls back to independent randomness each time
             (the old behaviour), controlled by n_reps / eval_reps.
    crn_seed: optional seed to make the CRN seed list itself reproducible
              across separate calls (so two runs with crn_seed=123 will
              search using literally the same underlying random streams).

    Returns: (p_star, pi_m_star, best_revenue, best_revenue_rate, history)
    history is a list of (p, revenue_rate) tuples evaluated along the way,
    useful for plotting / sanity-checking unimodality.
    """
    gr = (5 ** 0.5 - 1) / 2  # golden ratio conjugate ~= 0.618

    a, b = p_low, p_high
    history = []

    # Generate one shared set of seeds for the whole search (CRN).
    crn_seeds = None
    if use_crn:
        rng = random.Random(crn_seed)  # separate generator, doesn't disturb global random state
        crn_seeds = [rng.randrange(1, 2**31 - 1) for _ in range(crn_reps)]

    # cache to avoid recomputing f at points we already evaluated
    cache = {}

    def f(p):
        # round the key slightly so floating point noise doesn't cause
        # duplicate simulation runs at (numerically) the same price
        key = round(p, 6)
        if key in cache:
            return cache[key]
        pi_m_star, avg_revenue, avg_revenue_rate = evaluate_price(
            p, lambda1, lambda2, q, gamma, termination_time,
            n_reps=n_reps, eval_reps=eval_reps, verbose=False,
            crn_seeds=crn_seeds)
        cache[key] = (avg_revenue_rate, pi_m_star, avg_revenue)
        history.append((p, avg_revenue_rate))
        if verbose:
            print(f"  eval p={p:7.3f} -> pi_m*={pi_m_star:.4f}, "
                  f"revenue={avg_revenue:8.2f}, revenue_rate={avg_revenue_rate:.4f}")
        return cache[key]

    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)[0]
    fd = f(d)[0]

    for it in range(max_iter):
        if verbose:
            print(f"iter {it:3d}: a={a:.4f}, b={b:.4f}, interval={b - a:.4f}")

        if fc < fd:
            # maximum lies in [c, b]
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)[0]
        else:
            # maximum lies in [a, d]
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)[0]

        if abs(b - a) < tol:
            if verbose:
                print(f"Converged after {it + 1} iterations, interval width={b - a:.6f}")
            break

    p_star = (a + b) / 2
    revenue_rate_star, pi_m_star, revenue_star = f(p_star)

    return p_star, pi_m_star, revenue_star, revenue_rate_star, history


# 5: Repeat the whole golden section search several independent times and
#    summarize the resulting p* with a mean and a confidence interval.
#    This directly answers "how much does my optimal price estimate vary,
#    and what's my best single number".
def repeated_golden_section_search(lambda1, lambda2, q, gamma, termination_time,
                                    p_low, p_high, n_repeats=10,
                                    tol=1e-2, max_iter=50,
                                    n_reps=20, eval_reps=None,
                                    use_crn=True, crn_reps=30,
                                    verbose=False):
    """
    Run golden_section_search n_repeats times (each with an independent
    CRN seed list, i.e. independent "randomness draws"), collect the p*
    from each run, and report mean, std, and a 95% confidence interval.

    Returns: (p_mean, p_std, ci95_halfwidth, p_star_list)
    """
    import statistics

    p_list = []
    for r in range(n_repeats):
        p_star, pi_m_star, revenue_star, revenue_rate_star, _ = golden_section_search(
            lambda1, lambda2, q, gamma, termination_time,
            p_low=p_low, p_high=p_high, tol=tol, max_iter=max_iter,
            n_reps=n_reps, eval_reps=eval_reps,
            use_crn=use_crn, crn_reps=crn_reps,
            crn_seed=None,   # each repeat gets its own fresh random CRN set
            verbose=False)
        p_list.append(p_star)
        if verbose:
            print(f"repeat {r + 1:2d}/{n_repeats}: p* = {p_star:.3f} "
                  f"(revenue_rate={revenue_rate_star:.4f})")

    p_mean = statistics.mean(p_list)
    p_std = statistics.stdev(p_list) if len(p_list) > 1 else 0.0
    # 95% CI half-width using normal approximation (t-distribution would be
    # more correct for small n_repeats, but this is a reasonable default)
    ci95_halfwidth = 1.96 * p_std / (len(p_list) ** 0.5) if len(p_list) > 1 else 0.0

    print(f"\n{n_repeats} independent searches -> "
          f"p* mean = {p_mean:.3f}, std = {p_std:.3f}, "
          f"95% CI = [{p_mean - ci95_halfwidth:.3f}, {p_mean + ci95_halfwidth:.3f}]")

    return p_mean, p_std, ci95_halfwidth, p_list


if __name__ == "__main__":
    # parameters
    lambda1 = 1
    lambda2 = 2
    q = 0.1
    gamma = 0.2
    termination_time = 1000
    n_reps = 20

    # ---- Step 1: sanity check the fixed point solve at a fixed price ----
    p = 20
    pi_m_star, n_iter = fixed_point_solve(
        p, lambda1, lambda2, q, gamma, termination_time,
        tol=1e-4, max_iter=100, n_reps=n_reps, verbose=True)

    print(f"\n pi_m* = {pi_m_star:.4f}"
          f"({n_iter}iterations, price p={p})")

    # After convergence, run n_reps more simulations at pi_m_star to report
    # the platform's average revenue and average revenue rate under this price p
    total_revenue = 0.0
    total_revenue_rate = 0.0
    for _ in range(n_reps):
        _, _, _, _, revenue, revenue_rate = simulate(
            pi_m_star, p, lambda1, lambda2, q, gamma, termination_time)
        total_revenue += revenue
        total_revenue_rate += revenue_rate

    avg_revenue = total_revenue / n_reps
    avg_revenue_rate = total_revenue_rate / n_reps
    print(f" avg_revenue = {avg_revenue:.2f}, avg_revenue_rate = {avg_revenue_rate:.4f}"
          f" (at pi_m*={pi_m_star:.4f}, price p={p})")

    # ---- Step 2: golden section search over price to maximize revenue rate ----
    # use_crn=True: every candidate price is tested against the SAME batch
    # of random seeds within this one search, so the comparisons that drive
    # interval shrinking are far less noisy -> single-run result is already
    # more stable than before.
    print("\n=== Golden section search for optimal price (with CRN) ===")
    p_star, pi_m_at_star, revenue_star, revenue_rate_star, history = golden_section_search(
        lambda1, lambda2, q, gamma, termination_time,
        p_low=0, p_high=100, tol=1.0,      # tol=1.0 means price resolved to within ~1 unit
        n_reps=n_reps, eval_reps=40,       # bump up eval_reps for a less noisy final estimate
        use_crn=True, crn_reps=40,
        verbose=True)

    print(f"\n Optimal price p* = {p_star:.3f}")
    print(f" pi_m*(p*)         = {pi_m_at_star:.4f}")
    print(f" avg_revenue        = {revenue_star:.2f}")
    print(f" avg_revenue_rate   = {revenue_rate_star:.4f}")

    # ---- Step 3: repeat the search several times to see how much p*
    #      still varies, and get a mean +/- confidence interval ----
    print("\n=== Repeating the search 10 times for a confidence interval ===")
    p_mean, p_std, ci95_halfwidth, p_list = repeated_golden_section_search(
        lambda1, lambda2, q, gamma, termination_time,
        p_low=0, p_high=100, n_repeats=10,
        tol=1.0, n_reps=n_reps, eval_reps=40,
        use_crn=True, crn_reps=40, verbose=True)
    print(f" all p* found: {[round(x, 2) for x in p_list]}")
