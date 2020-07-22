import numpy as np
import paper_real_data.load as load


def ucb1(arms, mu, K: int, T: int) -> np.ndarray:
    """Implementazione algoritmo UCB1 con dati reali

    :param arms: lista degli 'asin' per ogni braccio
    :param mu: vettore delle medie per ogni braccio
    :param K: numero di braccia
    :param T: orizzonte temporale
    :return: regret
    """
    arms_pulls = np.zeros(K, dtype=int)
    pull_number = np.ones(K, dtype=int)
    average_reward = np.zeros(K, dtype=float)
    datas = []
    for i in range(0, K):
        datas.append(load.df.loc[load.df['asin'] == arms[i], 'reward'])
    for i in range(0, K):
        average_reward[i] = (datas[i].iloc[[arms_pulls[i]]])
        arms_pulls[i] += 1

    for t in range(K, T):
        UCB = average_reward + np.sqrt(2 * np.log(T) / pull_number)
        pos = np.argmax(UCB, 0).astype(int)
        weight = 1 / (pull_number[pos] + 1)
        arms_pulls[pos] += 1
        average_reward[pos] = (1 - weight) * average_reward[pos] + \
                             weight * \
                              (datas[pos].iloc[[arms_pulls[pos]]])
        pull_number[pos] += 1

    regret = np.dot((mu[0]-mu), pull_number)
    return regret
