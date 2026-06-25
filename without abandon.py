import random

# generate a random number with exponential distribution
def generate_exp(rate):
    return random.expovariate(rate)

# set parameters
lambda1=5
lambda2=4
q=0.3 
termination_time=1000 
seed=None
    
# initialize Q
Q1 = 0
Q2 = 0

# initialize EA
event_calendar = [
    generate_exp(lambda1),# first customer also follow exponential distribution
    generate_exp(lambda2),
    termination_time,
    ]

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
            else:
                Q2 = Q2 - 1

            # schedule next arrival time of Q1
            event_calendar[0] = t + generate_exp(lambda1)

        # Second col
        elif col == 1:
            u = random.uniform(0, 1)
            if u < (1 - q) ** Q1:
                Q2 = Q2 + 1
            else:
                Q1 = Q1 - 1

            event_calendar[1] = t + generate_exp(lambda2)

        # Third col
        else:
            break

print(f"{Q1}")
print(f"{Q2}")


