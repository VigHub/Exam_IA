from agents_simple_case import random_agent
from agents_simple_case import epsilon_greedy_agent
from agents_simple_case import thompson_agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    reward_prob_list = [0.2, 0.6, 0.7, 0.4, 0.6, 0.1]
    rounds = 100
    steps = 1000
    lines = "\n----------------------------\n"
    rnd = random_agent.play(reward_prob_list, rounds, steps)  # agente casuale
    print(lines)
    eg = epsilon_greedy_agent.play(reward_prob_list, rounds, steps)  # agente epsilon-greedy
    print(lines)
    th = thompson_agent.play(reward_prob_list, rounds, steps)  # agente di thompson
    print(lines)
    best_theorical = max(reward_prob_list) * steps  # massimo ottenibile sapendo a priori le probabilità

    agents = ["Thompson", "Random", "Epsilon-Greedy", "Onniscente"]
    plt.bar(agents, [th, rnd, eg, best_theorical])
    plt.title(f"Ricompensa media per diversi agenti in {rounds} round da {steps} passi ciascuno")
    plt.show()
