import random, math, time
import rl
import numpy as np
import matplotlib.pyplot as plt


def rand():
    return random.uniform(0,1)

def get_floor(probabilities, excluding=None):
    """Select a floor. If we must exclude one, it's an optional parameter"""
    prob = probabilities[:]
    floor_prob = rand()

    # If we are excluding a floor, remove it from the list
    if excluding is not None:
        prob.pop(excluding)
        prob = [i/sum(prob) for i in prob]

    # Perform a basic proportional search for a floor
    for i, floor in enumerate(prob):
        if floor > floor_prob:
            # If we're excluding a floor here, we need to return the index including the excluded
            if excluding is None or i < excluding:
                return i
            return i + 1
        floor_prob -= floor


def get_requestor(E, L, i=None):
    """Little recursive lambda generator which returns a tuple of selected floors according to E, L"""
    if i != None:
        return (i, get_floor(L, i))
    return lambda: get_requestor(E, L, get_floor(E))


def generate_vectors(K):
    """Given K floors, generate vectors E,L,P which have bottom heavy exits (L)"""
    E = [rand() for i in range(K)]
    E = [i/sum(E) for i in E]
    E.insert(0, 0)

    L = [rand() for i in range(K-1)]
    L.insert(0, sum(L)*10)
    L = [i/sum(L) for i in L]
    L.insert(0, 0)

    # Initial probability
    P = [1 for i in range(K)]
    P = [float(i)/sum(P) for i in P]
    P.insert(0, 0)

    return (E, L, P)


def waiting_time(optimal, request):
    h1 = np.random.normal(0, 0.01, 1)[0]
    h2 = np.random.normal(0, 0.01, 1)[0]
    f_i = 0.8 * optimal + 0.4 * math.ceil(optimal / 2) + h1
    f_req = 0.8 * request + 0.4 * math.ceil(request / 2) + h2
    return abs(f_req - f_i)


def environment(idle, request, K):
    """Returns penalty or not"""
    time = waiting_time(idle, request)
    worst_time = waiting_time(1, K)
    return time >= worst_time/2


def LRI():

    """What if there's one machine"""
    g = rl.lri_g
    f = lambda action, penalty, P: rl.lri_f(action, penalty, P, reward_const)
    reward_const = 0.005
    experiment = 100
    accuracy = 0
    overall_time = 0
    ave_time = []
    total_time = 0
    speed = []
    # absorbing_simulation
    for _ in range(experiment):
        K=6
        V = generate_vectors(K)
        E = V[0]
        L = V[1]
        P = V[0]
        accuracy_count = [0, 0, 0, 0, 0, 0, 0]
        P = P[:]
        requestor = get_requestor(E, L)
        check_penalty = lambda idle, request, K: environment(idle, request, K)
        current_floor = 1
        count = 0
        speed = []
        overall_time = 0

        for i in range(10000):
            optimal_floor = g(P)
            request = requestor()
            penalty = check_penalty( optimal_floor, request[0], K)
            P = f(optimal_floor, penalty, P)
            # print(P)
            count += 1
        start_time = time.time()
        for i in range(1000):
            speed.append(time.time() - start_time)
            optimal_floor = g(P)
            request = requestor()
            WT = waiting_time(optimal_floor, request[0])
            overall_time += WT
            accuracy_count[optimal_floor] += 1
            penalty = check_penalty(optimal_floor, request[0], K)
            P = f(optimal_floor, penalty, P)
            count += 1
            # print(accuracy_count)
        accuracy += max(accuracy_count)
        ave_time.append(overall_time/1000)
    plt.plot([i for i in range(experiment)], ave_time)
    plt.show()
    total_ave_time = sum(ave_time)/100
    print(total_ave_time)
    return (accuracy/experiment)/1000.0, sum(speed)/1000


def tsettlin():
    # Testing Tsettlin automata
    ave_time = []
    speed = []
    experiments = 100
    track = 1000
    accuracy = 0
    N = 5
    K = 6
    f_t = lambda p, s: rl.tsettlin_f(p, s, N)
    g_t = lambda s: rl.tsettlin_g(s, N)
    for _ in range(experiments):
        accuracy_count = [0, 0, 0, 0, 0, 0, 0]
        current_floor = 1
        state = current_floor*N
        V = generate_vectors(K)
        E = V[0]
        L = V[1]
        overall_time = 0
        requestor = get_requestor(E, L)
        start_time = time.time()
        for i in range(11000):
            request = requestor()
            action = g_t(state)
            if i >= 10000:
                speed.append(time.time() - start_time)
                WT = waiting_time(action, request[0])
                overall_time += WT
            accuracy_count[action] += 1
            # print(accuracy_count)
            penanty = environment(action, request[0], K)
            state = f_t(penanty, state)
            current_floor = request[0]
        ave_time.append(overall_time / track)
        accuracy += max(accuracy_count)
    plt.plot([i for i in range(experiments)], ave_time)
    plt.show()
    return (accuracy/11000)/experiments, sum(ave_time)/experiments, (sum(speed)/experiments)/1000

def krinsky():
    ave_time = []
    speed = []
    experiments = 100
    track = 1000
    accuracy = 0
    N = 5
    K = 6
    f_t = lambda p, s: rl.krinsky_f(p, s, N)
    g_t = lambda s: rl.tsettlin_g(s, N)
    for _ in range(experiments):
        accuracy_count = [0, 0, 0, 0, 0, 0, 0]
        current_floor = 1
        state = current_floor * N
        V = generate_vectors(K)
        E = V[0]
        L = V[1]
        overall_time = 0
        requestor = get_requestor(E, L)
        start_time = time.time()
        for i in range(11000):
            request = requestor()
            action = g_t(state)
            if i >= 10000:
                speed.append(time.time() - start_time)
                WT = waiting_time(action, request[0])
                overall_time += WT
            accuracy_count[action] += 1
            # print(accuracy_count)
            penanty = environment(action, request[0], K)
            state = f_t(penanty, state)
            current_floor = request[0]
        ave_time.append(overall_time / track)
        accuracy += max(accuracy_count)
    plt.plot([i for i in range(experiments)], ave_time)
    plt.show()
    return (accuracy / 11000) / experiments, sum(ave_time) / experiments, (sum(speed) / experiments) / 1000


def krylov():
    ave_time = []
    speed = []
    experiments = 100
    track = 1000
    accuracy = 0
    N = 5
    K = 6
    f_t = lambda p, s: rl.krylov_f(p, s, N)
    g_t = lambda s: rl.tsettlin_g(s, N)
    for _ in range(experiments):
        accuracy_count = [0, 0, 0, 0, 0, 0, 0]
        current_floor = 1
        state = current_floor * N
        V = generate_vectors(K)
        E = V[0]
        L = V[1]
        overall_time = 0
        requestor = get_requestor(E, L)
        start_time = time.time()
        for i in range(11000):
            request = requestor()
            action = g_t(state)
            if i >= 10000:
                speed.append(time.time() - start_time)
                WT = waiting_time(action, request[0])
                overall_time += WT
            accuracy_count[action] += 1
            # print(accuracy_count)
            penanty = environment(action, request[0], K)
            state = f_t(penanty, state)
            current_floor = request[0]
        ave_time.append(overall_time / track)
        accuracy += max(accuracy_count)
    plt.plot([i for i in range(experiments)], ave_time)
    plt.show()
    return (accuracy / 11000) / experiments, sum(ave_time) / experiments, (sum(speed) / experiments) / 1000


# print(LRI())
# print(tsettlin())
# print(krinsky())
print(krylov())


