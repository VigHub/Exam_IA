from agents_simple_case.multi_armed_bandit import MultiArmedBandit
import settings
import numpy as np


def get_random_choice(possible_choices: int):
    """Scelta dell'agente casuale.

    :param possible_choices: insieme di leve da cui effettuare la scelta
    :return: leva da abbassare
    """
    return np.random.randint(0, possible_choices)


def play(reward_prob_list: list, rounds: int, steps: int):
    """Aziona l'agente casuale

    :param reward_prob_list: insieme di braccia con probabilità per crare MultiArmedBandit
    :param rounds: numero di volte in cui viene ripetuto l'esperimento
    :param steps: numero di turni (scelte) da fare in ogni round
    :return: reward medio ottenuto per round
    """
    bandit = MultiArmedBandit(reward_prob_list)
    arms = len(bandit.reward_prob_list)
    reward_list = []
    reward_counter_arm = np.zeros(arms)
    counter_arm = np.zeros(arms)
    print("Inizio agente casuale")
    for i in settings.progress_bar(range(0, rounds)):
        tot_reward = 0
        for step in range(steps):
            choice = get_random_choice(arms)
            reward = bandit.step(choice)
            reward_counter_arm[choice] += reward
            counter_arm[choice] += 1
            tot_reward += reward
        reward_list.append(tot_reward)
    mean = np.mean(reward_list)
    print(f"Ricompensa totale media: {mean}")
    print(f"Probabilità ottenuta per braccio: {np.true_divide(reward_counter_arm, counter_arm)}")
    print(f"Probabilità \"reale\" per braccio {bandit.reward_prob_list}")
    return mean
