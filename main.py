import randon_agent
import epsilon_greedy_agent
import thompson_agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    reward_prob_list = [0.2, 0.6, 0.7, 0.4, 0.6, 0.1]
    rounds = 100
    steps = 1000
    lines = "\n----------------------------\n"
    rnd = randon_agent.play(reward_prob_list, rounds, steps)
    print(lines)
    eg = epsilon_greedy_agent.play(reward_prob_list, rounds, steps)
    print(lines)
    th = thompson_agent.play(reward_prob_list, rounds, steps)
    print(lines)
    best_theorical= max(reward_prob_list) * steps

    agents = ["Thompson", "Random", "Epsilon-Greedy", "Onniscente"]
    plt.bar(agents, [th, rnd, eg, best_theorical])
    plt.title(f"Ricompensa media per diversi agenti in {rounds} round da {steps} passi ciascuno")
    plt.show()
