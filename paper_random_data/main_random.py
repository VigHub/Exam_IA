import numpy as np
import matplotlib.pyplot as plt
from paper_random_data.ucb1 import ucb1
from paper_random_data.BASEFunc import BASEFunc
from paper_random_data.PRCS_twoarm import PRCS_twoarm
import math
import settings

K = 3
T = 50000
M = 3
gamma = 0.5
iterations = 200
K_set = np.round(np.logspace(math.log(2, 10), math.log(20, 10), endpoint=True, num=6)).astype(int)
T_set = np.round(np.logspace(math.log(500, 10), math.log(T, 10), endpoint=True, num=6)).astype(int)
M_set = np.arange(2, 10).astype(int)
mu_best = 0.6
mu_others = 0.5

# set True if you want to do that analysis
m1 = True  # M batches
k1 = True  # K armes
t1 = True  # T horizon
c1 = True  # comparison BaSE & ETC


# dependence on M
if m1:
    p = len(M_set)
    reg_minimax_M = np.zeros([iterations, p])
    reg_arithmetic_M = np.zeros([iterations, p])
    reg_geometric_M = np.zeros([iterations, p])
    reg_ucb1_M = np.zeros(iterations)
    mu = np.full(K, mu_others)
    mu[0] = mu_best
    print('Start on M')
    for i in settings.progress_bar(range(0, iterations)):
        reg_ucb1_M[i] = ucb1(mu, K, T)
        for m in range(0, len(M_set)):
            m_now = M_set[m]
            reg_minimax_M[i, m] = BASEFunc(mu, K, T, m_now, 'minimax', gamma)[0]
            reg_arithmetic_M[i, m] = BASEFunc(mu, K, T, m_now, 'arithmetic', gamma)[0]
            reg_geometric_M[i, m] = BASEFunc(mu, K, T, m_now, 'geometric', gamma)[0]

    mean_reg_minimax_M = np.mean(reg_minimax_M, 0) / T
    mean_reg_arithmetic_M = np.mean(reg_arithmetic_M, 0) / T
    mean_reg_geometric_M = np.mean(reg_geometric_M, 0) / T
    mean_reg_ucb1_M = np.mean(reg_ucb1_M, 0) / T

    plt.figure(0)
    plt.plot(M_set, mean_reg_minimax_M, label='minimax', marker='s')
    plt.plot(M_set, mean_reg_arithmetic_M, label='arithmetic', marker='<', linestyle='dashdot')
    plt.plot(M_set, mean_reg_geometric_M, label='geometric', marker='o', linestyle='dashed')
    plt.plot(M_set, mean_reg_ucb1_M * np.ones(len(M_set)), label='ucb1', marker='d', linestyle='dotted')
    plt.legend()
    plt.xlabel('M')
    plt.ylabel('Average Regret')
    plt.title('Average Regret vs. M')
    plt.show()
#
# # dependence on K
if k1:
    p = len(K_set)
    reg_minimax_K = np.zeros([iterations, p])
    reg_arithmetic_K = np.zeros([iterations, p])
    reg_geometric_K = np.zeros([iterations, p])
    reg_ucb1_K = np.zeros([iterations, p])

    print('Start on K')
    for i in settings.progress_bar(range(0, iterations)):

        for k in range(0, p):
            mu = np.full(K_set[k], 0.5)
            mu[0] = 0.6
            reg_ucb1_K[i, k] = ucb1(mu, K_set[k], T)
            reg_minimax_K[i, k] = BASEFunc(mu, K_set[k], T, M, 'minimax', gamma)[0]
            reg_arithmetic_K[i, k] = BASEFunc(mu, K_set[k], T, M, 'arithmetic', gamma)[0]
            reg_geometric_K[i, k] = BASEFunc(mu, K_set[k], T, M, 'geometric', gamma)[0]

    mean_reg_minimax_K = np.mean(reg_minimax_K, 0) / T
    mean_reg_arithmetic_K = np.mean(reg_arithmetic_K, 0) / T
    mean_reg_geometric_K = np.mean(reg_geometric_K, 0) / T
    mean_reg_ucb1_K = np.mean(reg_ucb1_K, 0) / T

    plt.figure(1)
    plt.plot(K_set, mean_reg_minimax_K, label='minimax', marker='s')
    plt.plot(K_set, mean_reg_arithmetic_K, label='arithmetic', marker='<', linestyle='dashdot')
    plt.plot(K_set, mean_reg_geometric_K, label='geometric', marker='o', linestyle='dashed')
    plt.plot(K_set, mean_reg_ucb1_K, label='ucb1', marker='d', linestyle='dotted')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Average Regret')
    plt.title('Average Regret vs. K')

    plt.show()

# dependence on T
if t1:
    p = len(T_set)
    reg_minimax_T = np.zeros([iterations, p])
    reg_arithmetic_T = np.zeros([iterations, p])
    reg_geometric_T = np.zeros([iterations, p])
    reg_ucb1_T = np.zeros([iterations, p])
    mu = np.full(K, mu_others)
    mu[0] = mu_best

    print('Start on T')
    for i in settings.progress_bar(range(0, iterations)):
        for t in range(0, p):
            t_now = T_set[t]
            reg_minimax_T[i, t] = BASEFunc(mu, K, t_now, M, 'minimax', gamma)[0] / t_now
            reg_arithmetic_T[i, t] = BASEFunc(mu, K, t_now, M, 'arithmetic', gamma)[0] / t_now
            reg_geometric_T[i, t] = BASEFunc(mu, K, t_now, M, 'geometric', gamma)[0] / t_now
            reg_ucb1_T[i, t] = ucb1(mu, K, t_now) / t_now

    mean_reg_minimax_T = np.mean(reg_minimax_T, 0)
    mean_reg_arithmetic_T = np.mean(reg_arithmetic_T, 0)
    mean_reg_geometric_T = np.mean(reg_geometric_T, 0)
    mean_reg_ucb1_T = np.mean(reg_ucb1_T, 0)

    plt.figure(2)
    plt.semilogx(T_set, mean_reg_minimax_T, label='minimax', marker='s')
    plt.semilogx(T_set, mean_reg_arithmetic_T, label='arithmetic', marker='<', linestyle='dashdot')
    plt.semilogx(T_set, mean_reg_geometric_T, label='geometric', marker='o', linestyle='dashed')
    plt.semilogx(T_set, mean_reg_ucb1_T, label='ucb1', marker='d', linestyle='dotted')
    plt.legend()
    plt.xlabel('T')
    plt.ylabel('Average Regret')
    plt.title('Average Regret vs. T')
    plt.show()


# compare with [PRCS16]
if c1:
    p = len(M_set)
    reg_minimax = np.zeros([iterations, p])
    reg_geometric = np.zeros([iterations, p])
    reg_prcs_minimax = np.zeros([iterations, p])
    reg_prcs_geometric = np.zeros([iterations, p])
    reg_ucb1 = np.zeros(iterations)
    mu = np.array([mu_best, mu_others])
    print('Start on PRCS16')
    for i in settings.progress_bar(range(0, iterations)):
        reg_ucb1[i] = ucb1(mu, 2, T)
        for m in range(0, p):
            m_now = M_set[m]
            reg_minimax[i, m] = BASEFunc(mu, 2, T, m_now, 'minimax', gamma)[0]
            reg_geometric[i, m] = BASEFunc(mu, 2, T, m_now, 'geometric', gamma)[0]
            reg_prcs_minimax[i, m] = PRCS_twoarm(mu, m_now, T, 'minimax')
            reg_prcs_geometric[i, m] = PRCS_twoarm(mu, m_now, T, 'geometric')

    mean_reg_minimax = np.mean(reg_minimax, 0) / T
    mean_reg_geometric = np.mean(reg_geometric, 0) / T
    mean_reg_prcs_minimax = np.mean(reg_prcs_minimax, 0) / T
    mean_reg_prcs_geometric = np.mean(reg_prcs_geometric, 0) / T
    mean_reg_ucb1 = np.mean(reg_ucb1, 0) / T

    plt.figure(4)
    plt.plot(M_set, mean_reg_minimax, label='BaSE minimax', color='blue', marker='s')
    plt.plot(M_set, mean_reg_geometric, label='BaSE geometric', color='blue', marker='<', linestyle='dashdot')
    plt.plot(M_set, mean_reg_prcs_minimax, label='ETC minimax', color='red', marker='o', )
    plt.plot(M_set, mean_reg_prcs_geometric, label='ETC geometric', color='green', marker='o', linestyle='dashdot')
    plt.plot(M_set, mean_reg_ucb1 * np.ones(p), label='ucb1', marker='d', linestyle='dotted')
    plt.legend()
    plt.xlabel('M')
    plt.ylabel('Average Regret')
    plt.title('Average Regret BaSE vs. ETC')

    plt.show()
