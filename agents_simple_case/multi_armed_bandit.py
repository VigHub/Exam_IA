import numpy as np


class MultiArmedBandit:
    """Framework multi armed bandit """
    def __init__(self, reward_prob_list):
        """ Crea un nuovo sistema, con i parametri indicati

        :param reward_prob_list: probabilità effettiva di ottenere la ricompensa da un determinato braccio/leva
        """
        self.reward_prob_list = reward_prob_list

    def step(self, choice: int):
        """ | Abbassa il braccio selezionato
        | (reward è 1 o 0) => Bernoulli

        :param choice: il braccio da abbassare
        :return: risultato dovuto all'aver abbassato quella leva
        """
        if choice > len(self.reward_prob_list) or choice < 0:
            raise Exception(f"Errore, la scelta {choice} non è possibile. Possibili scelte: {self.reward_prob_list}")
        success = self.reward_prob_list[choice]
        failure = 1.0 - success
        return np.random.choice(2, p=[failure, success])
