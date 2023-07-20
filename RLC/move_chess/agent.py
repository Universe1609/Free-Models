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

    #Aplicamos la política en un estado para determinar la accion a tomar
    #esto es a lo que llamamos politica epsilon-greedy
    def apply_policy(self, state, epsilon):
        #encontramos la politica del agente ,que maximiza la recompensa, para el estado dado, esta es la accion "codiciosa"
        greedy_action_value = np.max(self.policy[state[0], state[1], :])
        #alamcenamos las acciones con el valor maximo 
        greedy_indices = [i for i, a in enumerate(self.policy[state[0], state[1], :]) if
                          a == greedy_action_value]
        #si encontramos mas de una accion con el mismo valor, elegimos aleatoriamente el valor.
        action_index = np.random.choice(greedy_indices)
        #si el valor aletorio es menor a eps, ya no elegimos la accion el valor maximo
        if np.random.uniform(0, 1) < epsilon:
            action_index = np.random.choice(range(len(self.action_space)))
        return action_index

    #comparan las politicas
    def compare_policies(self):
        return np.sum(np.abs(self.policy - self.policy_prev))

    #espacio de acciones del agente en función del tipo de pieza de ajedrez que representa
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
        elif self.piece == 'queen':
            self.action_space = []
            # movimientos diagonales como un alfil
            for amplitude in range(1, 8):
                self.action_space.append((-amplitude, amplitude))    
                self.action_space.append((amplitude, amplitude))    
                self.action_space.append((amplitude, -amplitude))   
                self.action_space.append((-amplitude, -amplitude))  
            # movimientos lineales como una torre
            for amplitude in range(1, 8):
                self.action_space.append((-amplitude, 0))  
                self.action_space.append((0, amplitude))  
                self.action_space.append((amplitude, 0))  
                self.action_space.append((0, -amplitude))  #
