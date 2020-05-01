from multi_armed_bandit import MultiArmedBandit
import numpy as np


def get_random_choice(possible_choices):
    return np.random.randint(0, possible_choices)


def play(reward_prob_list, rounds, steps):
    bandit = MultiArmedBandit(reward_prob_list)
    arms = len(bandit.reward_prob_list)
    reward_list = []
    reward_counter_arm = np.zeros(arms)
    counter_arm = np.zeros(arms)
    print("Inizio agente casuale")
    for i in range(0, rounds):
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
