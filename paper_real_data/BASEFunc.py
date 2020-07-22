import numpy as np
import math
import paper_real_data.load as load


def BASEFunc(arms, mu, K: int, T: int, M: int, grid_type: str, gamma: float) -> [float, np.ndarray]:
    """Implementazione algoritmo BaSE con dati reali

    :param arms: lista degli 'asin' per ogni braccio
    :param mu: vettore delle medie per ogni braccio
    :param K: numero di braccia
    :param T: orizzonte temporale
    :param M: numero di batch
    :param grid_type: tipologia di griglia
    :param gamma: parametro di tuning
    :return: [regret medio, insieme delle braccia attive nell'ultimo batch]
    """

    regret = 0
    datas = []
    for i in range(0, K):
        datas.append(load.df.loc[load.df['asin'] == arms[i], 'reward'])
    if grid_type == 'minimax':
        a = np.power(T, np.divide(1, np.subtract(2, np.power(2, 1.0 - M))))
        t_grid = np.floor(a ** (2 - 1 / 2 ** np.arange(0, M)))
        t_grid[M - 1] = T
        t_grid = np.insert(t_grid, 0, 0)
    elif grid_type == 'geometric':
        b = T ** (1.0 / M)
        t_grid = np.floor(b ** np.arange(1, M + 1))
        t_grid[M - 1] = T
        t_grid = np.insert(t_grid, 0, 0)
    elif grid_type == 'arithmetic':
        t_grid = np.floor(np.linspace(0, T, M+1))

    active_set = np.ones((K, 1), dtype=int)
    number_pull = np.zeros(K, dtype=int)
    average_reward = np.zeros(K, dtype=float)
    for i in range(1, M+1):
        available_k = np.sum(active_set)
        pull_number = int(np.round(np.maximum(np.floor((t_grid[i]-t_grid[i-1])/available_k), 1)))
        t_grid[i] = available_k * pull_number + t_grid[i-1]
        row_active, col_active = np.where(active_set == 1)
        for j in row_active:
            average_reward[j] = average_reward[j] * (number_pull[j]/(number_pull[j]+pull_number)) + \
                                (datas[j].iloc[number_pull[j]:(number_pull[j]+pull_number)]).mean() * \
                                (pull_number / (number_pull[j] + pull_number))
            regret += (pull_number * (mu[0] - mu[j]))
            number_pull[j] += pull_number

        row_active, col_active = np.where(active_set == 1)
        max_arm = np.max(average_reward[row_active])
        for j in row_active:
            if (max_arm - average_reward[j]) >= np.sqrt(gamma * math.log(K*T) / number_pull[j]):
                active_set[j] = 0

    return [regret, active_set]
