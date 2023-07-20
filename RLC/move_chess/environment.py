import pprint
import numpy as np


class Board(object):

    def __init__(self):
        self.state = (0, 0)
        self.reward_space = np.zeros(shape=(8, 8)) - 1
        self.terminal_state = (7, 5)

    def step(self, action):
        # Se obtiene la recompensa actual según la posición actual
        reward = self.reward_space[self.state[0], self.state[1]]
        # Comprobamos si el estado actual es el estado terminal
        if self.state == self.terminal_state:
            episode_end = True
            return 0, episode_end
        else:
            episode_end = False
            # Guarda el estado actual en una variable auxiliar
            old_state = self.state
            # Calcula el nuevo estado sumando las componentes de la acción al estado actual
            new_state = (self.state[0] + action[0], self.state[1] + action[1])
            # Si alguna coordenada del nuevo estado está fuera de los límites del tablero, se mantiene el estado actual; de lo contrario, se actualiza el estado actual al nuevo estado
            self.state = old_state if np.min(new_state) < 0 or np.max(new_state) > 7 else new_state
            return reward, episode_end

    def render(self):
        visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())
        # Marca la posición del estado actual como "[S]" en el tablero visual
        visual_board[self.state[0]][self.state[1]] = "[S]"
        # Marca la posición del estado terminal como "[F]" en el tablero visual
        visual_board[self.terminal_state[0]][self.terminal_state[1]] = "[F]"
        self.visual_board = visual_board
