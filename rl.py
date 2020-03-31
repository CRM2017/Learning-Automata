import random

def tsettlin_f(penalty, state, N):
    """State selection for Tsettlin"""
    if penalty:
        if state % N != 0:
            state = state + 1
        else: # swap state
            if state == N:
                state = 2 * N
            elif state == 2 * N:
                state = 3 * N
            elif state == 3 * N:
                state = 4 * N
            elif state == 4 * N:
                state = 5 * N
            elif state == 5 * N:
                state = 6 * N
            elif state == 6 * N:
                state = N
    else: # award
        if state % N != 1:
            state = state - 1

    return state


def tsettlin_g(state, N):
    """Action selection for tsettlin"""
    action = None
    if 1 <= state <= N:
        action = 1
    elif N+1 <= state <= 2*N:
        action = 2
    elif 2*N+1 <= state <= 3*N:
        action = 3
    elif 3*N+1 <= state <= 4*N:
        action = 4
    elif 4*N+1 <= state <= 5*N:
        action = 5
    elif 5*N+1 <= state <= 6*N:
        action = 6
    return action


def krinsky_f(penalty, state, N):
    """State selection for Krinsky"""
    if penalty:
        return tsettlin_f(penalty, state, N)
    if state <= N:
        return 1
    elif state <= 2*N:
        return N+1
    elif state <= 3*N:
        return 2*N+1
    elif state <= 4*N:
        return 3*N+1
    elif state <= 5*N:
        return 4*N+1
    elif state <= 6*N:
        return 5*N+1


def krylov_f(penalty, state, N):
    """State selection for Krylov"""
    if penalty and random.uniform(0, 1) <= 0.5:
        penalty = not penalty
    if penalty:
        if state % N != 0:  # In neither N nor 2N
            state = state + 1  # Weaken
        else:
            if state == N:
                state = 2 * N
            elif state == 2 * N:
                state = 3 * N
            elif state == 3 * N:
                state = 4 * N
            elif state == 4 * N:
                state = 5 * N
            elif state == 5 * N:
                state = 6 * N
            elif state == 6 * N:
                state = N
    else: # reward
        state = tsettlin_f(penalty, state, N)
    return state



def lri_g(P):
    """Select the action"""
    rng = random.uniform(0,1)


    for i, action in enumerate(P):
        if action > rng:
            return i
        rng -= action
    return len(P) - 1


def lri_f(action, penalty, P, r):
    # if penalty, do nothing
    P = P[:] # eg. P = [0.1,0.2,0.3,0.4]
    if penalty:
        return P

    identity = [ int(i == action) for i, _ in enumerate(P)]  # eg. P = [0,0,1,1]
    identity = [i * (r) for i in identity]  # eg. [0.0, 0.0, 0.2, 0.0] (assume r=0.2)

    # Multiply each element in the P vector by 1-r
    P = [i * (1-r) for i in P]  # eg. [0.08, 0.16, 0.24, 0.32]

    # Add the two vectors together
    P = [sum(a) for a in zip(identity, P)] # eg. [0.08, 0.16, 0.44, 0.32]
    return P

