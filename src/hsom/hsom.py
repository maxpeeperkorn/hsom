import numpy as np


def habituate(w, S, tau=3.33, alpha=1.05, w_0=1.0):
    """ Returns a value for how the synapse weight should change.

        :param w        current synaptic weights
        :param S        synaptic stimulus (1 for falling efficacy, 0 for rising efficacy)
        :param tau      habituation rate
        :param alpha    recovery rate
        :param w_0      initial synaptic weights
        :return         change for the new value of the synaptic weight

        Example use:
        w += habituate(w, S)

        w, S tau, alpha and w_0 could presented as single numbers are as equal size matrices.
    """
    return (alpha * (w_0 - w) - S) / tau

def euclidean_distance(x, w):
    return np.linalg.norm(w - x, axis=-1) 

def learn(x, w, learning_rate=.25, mask=1):
    return learning_rate * mask * (x - w)


class HSOM:
    def __init__(self, grid_size, n_inputs, n_outputs, learning_rate=.25, 
                 habituation_rates=[3.33, 14.3], recovery_rate=1.05, forgetting=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
      
        self.features = np.random.normal(0, 1, (grid_size, grid_size, n_inputs))
        self.synapses = np.ones(shape=(grid_size, grid_size))

        self.learning_rate = learning_rate
        self.habituation_rates = habituation_rates
        self.recovery_rate = recovery_rate
        self.forgetting = forgetting

    def get_features(self):
        return self.features

    def get_features(self):
        return self.synapses

    def neighbourhood(self, winner):
        """ Returns a array mask for the winner and its topological neighbours. """
        mask = np.zeros(shape=(self.grid_size, self.grid_size))
        _x, _y = winner

        if _x >= self.grid_size or _x < 0 or _y >= self.grid_size or _y < 0:
            raise ValueError("Winner is out of bounds.")

        for x in range(_x - 1, _x + 2):
            if x >= 0 and x < self.grid_size:
                for y in range(_y - 1, _y + 2):
                    if y >= 0 and y < self.grid_size:
                        mask[x][y] = 1.0
        return mask

    def winner(self, distance_matrix):
        return np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

    def cycle(self, x):
        distance_matrix = euclidean_distance(x, self.features)
        winner = self.winner(distance_matrix)
        novelty = self.synapses[winner]

        # neighbourhood is a matrix of 0 and 1, 1 where the values should update
        neighbour_mask = self.neighbourhood(winner)
        
        # habituation rate matrix, maybe a third habituation rate for forgetting
        tau = np.ones(shape=(self.grid_size, self.grid_size)) * self.habituation_rates[1]
        tau[winner] = self.habituation_rates[0]

        self.features += learn(x, self.features, self.learning_rate, neighbour_mask)
        self.synapses += habituate(self.synapses, neighbour_mask, tau=tau, alpha=self.recovery_rate)
        
        return novelty, winner