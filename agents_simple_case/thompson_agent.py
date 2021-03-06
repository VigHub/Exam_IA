from agents_simple_case.multi_armed_bandit import MultiArmedBandit
import settings
import numpy as np


def get_thompson_choice(success_count_arm: np.ndarray, failure_count_arm: np.ndarray):
    """Scelta dell'agente di Thompson

    :param success_count_arm: numero di reward ottenuti per ogni braccio
    :param failure_count_arm: numero di volte in cui ogni braccio non ha dato nulla
    :return: leva da abbassare
    """
    beta_arr = np.random.beta(success_count_arm, failure_count_arm)
    return np.argmax(beta_arr)


def play(reward_prob_list: list, rounds: int, steps: int):
    """Aziona l'agente di Thompson

    :param reward_prob_list: insieme di braccia con probabilità per crare MultiArmedBandit
    :param rounds: numero di volte in cui viene ripetuto l'esperimento
    :param steps: numero di turni (scelte) da fare in ogni round
    :return: reward medio ottenuto per round
    """
    bandit = MultiArmedBandit(reward_prob_list)
    arms = len(bandit.reward_prob_list)
    tot_reward_list = []
    mean_prob_arm_list = np.zeros(arms)
    print("Inizio agente di Thompson")
    for i in settings.progress_bar(range(0, rounds)):
        tot_reward = 0
        success_count_arm = np.ones(arms)
        failure_count_arm = np.ones(arms)
        counter_arm = np.full(arms, 1.0e-6)
        for step in range(0, steps):
            choice = get_thompson_choice(success_count_arm, failure_count_arm)
            reward = bandit.step(choice)
            if reward == 1:
                success_count_arm[choice] += 1
            else:  # reward == 0
                failure_count_arm[choice] += 1
            counter_arm[choice] += 1
            tot_reward += reward
        tot_reward_list.append(tot_reward)
        prob_arm_list = np.true_divide(success_count_arm, counter_arm)
        mean_prob_arm_list += prob_arm_list
    mean = np.mean(tot_reward_list)
    print(f"Ricompensa totale media: {mean}")
    print(f"Probabilità ottenuta per braccio: {mean_prob_arm_list / rounds}")
    print(f"Probabilità \"reale\" per braccio {bandit.reward_prob_list}")
    return mean
