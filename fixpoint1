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
    return total_joined, total_abandoned, Q1, Q2

# 2: Fixpoint calculation
def fixed_point_solve(p, lambda1, lambda2, q, gamma, termination_time,
                       tol=1e-4, max_iter=100, n_reps=20, verbose=True):
    
    # initial pi_m value
    pi_m = 1.0

    # Perform max_iter iterations
    for k in range(max_iter):
        total_joined = 0
        total_abandoned = 0

        # Perform n_rep independent simulations for each iteration
        for _ in range(n_reps):
            joined, abandoned, Q1_end, Q2_end = simulate(
                pi_m, p, lambda1, lambda2, q, gamma, termination_time)
            # Calculate the total_joined and total_abandonedsum of n_rep simulations
            total_joined += joined
            total_abandoned += abandoned

        # Extreme situation: Price too high, no one joining
        if total_joined == 0:
            if verbose:
                print(f"{k}: No one joined in {n_reps}th repeated simulation, iteration stopped")
       
        # Calculate new pi_m
        pi_m_new = 1 - total_abandoned / total_joined
        diff = abs(pi_m_new - pi_m)

        # Print calculation details
        if verbose:
            print(f"iter {k:3d}: pi_m={pi_m:.6f} -> {pi_m_new:.6f} "
                  f"(Total joined={total_joined}, abandoned={total_abandoned}, "
                  f"n_reps={n_reps}, diff={diff:.6f})")
            
        # Update pi_m in the Kth iteration
        pi_m = pi_m_new
        
        # Determine whether the convergence requirements have been met
        if diff < tol:
            if verbose:
                print(f"Converge to the {k}th iteration")
            break

    return pi_m, k + 1


if __name__ == "__main__":
    # parameters
    lambda1 = 5
    lambda2 = 4
    q = 0.3
    gamma = 0.5
    termination_time = 1000
    p = 40  
    n_reps = 20  

    pi_m_star, n_iter = fixed_point_solve(
        p, lambda1, lambda2, q, gamma, termination_time,
        tol=1e-4, max_iter=100, n_reps=n_reps, verbose=True )

    print(f"\n pi_m* = {pi_m_star:.4f}"
          f"({n_iter}iterations, price p={p})")
