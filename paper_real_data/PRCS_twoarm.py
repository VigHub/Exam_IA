import math
import numpy as np
import paper_real_data.load as load


def PRCS_twoarm(arms, mu, M: int, T: int, grid_type: str) -> float:
    """Implementazione algoritmo ETC con dati reali

    :param arms: lista degli 'asin' per ogni braccio
    :param mu: vettore delle medie per ogni braccio
    :param T: orizzonte temporale
    :param M: numero di batch
    :param grid_type: tipologia di griglia
    :return: regret
    """
    datas = []
    for i in range(0, 2):
        datas.append(load.df.loc[load.df['asin'] == arms[i], 'reward'])
    if grid_type == 'minimax':
        # a = T ** (1 / (2 - 2 ** (1-M)))
        a = np.power(T, np.divide(1, np.subtract(2, np.power(2, 1.0 - M))))
        t_grid = np.floor(a ** (2 - 1 / 2 ** np.arange(0, M)))
        t_grid[M - 1] = T
        t_grid = np.insert(t_grid, 0, 0)
    elif grid_type == 'geometric':
        b = T ** (1.0 / M)
        t_grid = np.floor(b ** np.arange(1, M + 1))
        t_grid[M - 1] = T
        t_grid = np.insert(t_grid, 0, 0)

    pull_number = np.round(t_grid[1] / 2).astype(int)
    # il braccio migliore è il primo
    regret = pull_number * (mu[0] - mu[1])
    reward = np.array([
        datas[0].iloc[0: pull_number].sum(),
        datas[1].iloc[0: pull_number].sum()
    ])
    opt = 0

    for m in range(1, M):
        t = t_grid[m]
        threshold = math.sqrt(4 * math.log(2 * T / t) / t)
        if opt == 0:
            if (reward[0] - reward[1]) / t > threshold:
                opt = 1
            elif (reward[1] - reward[0]) / t > threshold:
                opt = 2
            else:
                cur_number = np.round((t_grid[m + 1] - t_grid[m]) / 2).astype(int)
                reward += np.array([
                    datas[0].iloc[pull_number: pull_number+cur_number].sum(),
                    datas[1].iloc[pull_number: pull_number+cur_number].sum()
                ])
                pull_number += cur_number
                regret += cur_number * (mu[0] - mu[1])
        if opt == 2:  # scelgo il secondo quindi ho regret
            regret += (t_grid[m + 1] - t_grid[m]) * (mu[0] - mu[1])
        if m == (M - 2):
            if reward[0] > reward[1]:
                opt = 1
            else:
                opt = 2
    return regret
