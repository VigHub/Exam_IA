from multi_armed_bandit import MultiArmedBandit
import numpy as np
import random as rnd


def get_epsilon_greedy_choice(epsilon, reward_counter_array):
    possible_choice = len(reward_counter_array)
    if rnd.uniform(0, 1) <= epsilon:
        choice = np.random.randint(0, possible_choice)
    else:
        max_arm = np.amax(reward_counter_array)
        indices = np.where(reward_counter_array == max_arm)[0]
        choice = np.random.choice(indices)
    return choice


def play(reward_prob_list, rounds, steps):
    bandit = MultiArmedBandit(reward_prob_list)
    espilon = 0.1
    arms = len(bandit.reward_prob_list)

    tot_reward_list = []
    mean_prob_arm_list = np.zeros(arms)
    print("Inizio agente epsilon-greedy")
    for i in range(0, rounds):
        tot_reward = 0
        reward_counter_arm = np.zeros(arms)
        counter_arm = np.full(arms, 1.0e-6)
        for step in range(0, steps):
            choice = get_epsilon_greedy_choice(espilon, reward_counter_arm)
            reward = bandit.step(choice)
            reward_counter_arm[choice] += reward
            counter_arm[choice] += 1
            tot_reward += reward
        tot_reward_list.append(tot_reward)
        prob_arm_list = np.true_divide(reward_counter_arm, counter_arm)
        mean_prob_arm_list += prob_arm_list
    mean = np.mean(tot_reward_list)
    print(f"Ricompensa totale media: {mean}")
    print(f"Probabilità ottenuta per braccio: {mean_prob_arm_list / rounds}")
    print(f"Probabilità \"reale\" per braccio {bandit.reward_prob_list}")
    return mean
