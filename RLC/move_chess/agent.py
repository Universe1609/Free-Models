import numpy as np
import pprint


class Piece(object):

    def __init__(self, piece='king'):
        self.piece = piece
        self.init_actionspace()
        self.value_function = np.zeros(shape=(8, 8))
        self.value_function_prev = self.value_function.copy()
        self.N = np.zeros(shape=(8, 8))
        self.E = np.zeros(shape=(8, 8))
        self.Returns = {}
        self.action_function = np.zeros(shape=(8, 8, len(self.action_space)))
        self.policy = np.zeros(shape=self.action_function.shape)
        self.policy_prev = self.policy.copy()

    def apply_policy(self, state, epsilon):
        greedy_action_value = np.max(self.policy[state[0], state[1], :])
        greedy_indices = [i for i, a in enumerate(self.policy[state[0], state[1], :]) if
                          a == greedy_action_value]
        action_index = np.random.choice(greedy_indices)
        if np.random.uniform(0, 1) < epsilon:
            action_index = np.random.choice(range(len(self.action_space)))
        return action_index

    def compare_policies(self):
        return np.sum(np.abs(self.policy - self.policy_prev))

    def init_actionspace(self):
        assert self.piece in ["king", "rook", "bishop",
                              "knight"], f"{self.piece} is not a supported piece try another one"
        if self.piece == 'king':
            self.action_space = [(-1, 0),  
                                 (-1, 1),  
                                 (0, 1),  
                                 (1, 1),  
                                 (1, 0),  
                                 (1, -1),  
                                 (0, -1),  
                                 (-1, -1),  
                                 ]
        elif self.piece == 'rook':
            self.action_space = []
            for amplitude in range(1, 8):
                self.action_space.append((-amplitude, 0))  
                self.action_space.append((0, amplitude))  
                self.action_space.append((amplitude, 0))  
                self.action_space.append((0, -amplitude))  
        elif self.piece == 'knight':
            self.action_space = [(-2, 1),  
                                 (-1, 2), 
                                 (1, 2),  
                                 (2, 1),
                                 (2, -1),   
                                 (1, -2),  
                                 (-1, -2),  
                                 (-2, -1)]  
        elif self.piece == 'bishop':
            self.action_space = []
            for amplitude in range(1, 8):
                self.action_space.append((-amplitude, amplitude))  
                self.action_space.append((amplitude, amplitude))  
                self.action_space.append((amplitude, -amplitude))  
                self.action_space.append((-amplitude, -amplitude))  # 
