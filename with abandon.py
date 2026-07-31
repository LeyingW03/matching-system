import random

# generate a random number with exponential distribution
def generate_exp(rate):
    return random.expovariate(rate)

# set parameters
lambda1 = 1
lambda2 = 2
q = 0.1
gamma = 0.2  # patience rate
termination_time = 1000
seed = None

# initialize Q
Q1 = 0
Q2 = 0

# initialize event_calendar (first three columns unchanged)
event_calendar = [
    generate_exp(lambda1), 
    generate_exp(lambda2), 
    termination_time,      
]

# initialize direction
direction = [0, 0, 0]

t = 0.0  # set clock

# main cycle
while t < termination_time:

    # find minimum value col
    tn = min(event_calendar)
    col = event_calendar.index(tn)

    # push the clock
    t = tn

    # First col
    if col == 0:
        u = random.uniform(0, 1)

        if u < (1 - q) ** Q2:
            Q1 = Q1 + 1

            # Add the patience deadline for this user and mark it as Q1
            event_calendar.append(t + generate_exp(gamma))
            direction.append(1)
        else:
            Q2 = Q2 - 1

            # find all the users in Q2
            q2_indices = [i for i, d in enumerate(direction) if d == 2]
            #Randomly select one from Q2
            leave_idx = random.choice(q2_indices)
            # delete the Q2 user's patience record
            del event_calendar[leave_idx]
            del direction[leave_idx]

        # schedule next arrival time of Q1
        event_calendar[0] = t + generate_exp(lambda1)

    # Second col
    elif col == 1:
        u = random.uniform(0, 1)

        if u < (1 - q) ** Q1:
            Q2 = Q2 + 1

            event_calendar.append(t + generate_exp(gamma))
            direction.append(2)
        else:
            Q1 = Q1 - 1

            q1_indices = [i for i, d in enumerate(direction) if d == 1]
            leave_idx = random.choice(q1_indices)
            del event_calendar[leave_idx]
            del direction[leave_idx]

        event_calendar[1] = t + generate_exp(lambda2)

    # Third col
    elif col == 2:
        break

    # Forth col and beyond:the user's patience run out and leave
    else:
        # Determine which queue the user belongs to 
        # and reduce the corresponding number 
        if direction[col] == 1:
            Q1 = Q1 - 1
        else:
            Q2 = Q2 - 1

        # delete the user's patience record
        del event_calendar[col]
        del direction[col]

print(f"{Q1}")
print(f"{Q2}")
