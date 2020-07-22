import numpy as np


def ucb1(mu: list[float], K: int, T: int) -> float:
    """Implementazione algoritmo UCB1 con dati casuali

    :param mu: vettore delle medie per ogni braccio
    :param K: numero di braccia
    :param T: orizzonte temporale
    :return: regret
    """

    pull_number = np.ones(K, dtype=int)
    average_reward = np.zeros(K, dtype=float)
    for i in range(0, K):
        average_reward[i] = mu[i] + np.random.randn()

    for t in range(K, T):
        UCB = average_reward + np.sqrt(2 * np.log(T) / pull_number)
        pos = np.argmax(UCB, 0)
        weight = 1 / (pull_number[pos] + 1)
        average_reward[pos] = (1 - weight) * average_reward[pos] + \
                              weight * (mu[pos] + np.random.randn())
        pull_number[pos] += 1

    regret = np.dot((mu[0]-mu), pull_number)
    return regret
