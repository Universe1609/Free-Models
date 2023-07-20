import numpy as np
import pprint


class Reinforce(object):

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def play_episode(self, state, max_steps=1e3, epsilon=0.1):
        # Configuración inicial del episodio        
        self.env.state = state # Configura el estado inicial del entorno con el estado proporcionado
        states = []
        actions = []
        rewards = []
        episode_end = False

        count_steps = 0
        while not episode_end:
            count_steps += 1
            states.append(state) # Agrega el estado actual a la lista de estados visitados
            action_index = self.agent.apply_policy(state, epsilon)  #Aplicamos la política del agente para elegir una acción
            action = self.agent.action_space[action_index] # Obtenemos la acción correspondiente al índice seleccionado
            actions.append(action_index) # Añadimos el índice de la acción a la lista de acciones tomadas
            reward, episode_end = self.env.step(action)   # Realizar un paso en el entorno y obtener la recompensa y el indicador de fin de episodi
            state = self.env.state
            rewards.append(reward)  #Agrega la recompensa obtenida a la lista de recompensas
            
            if count_steps > max_steps:
                episode_end = True

        return states, actions, rewards # Devuelve las listas de estados, acciones y recompensas obtenidas durante el episodio

    def sarsa_td(self, n_episodes=1000, alpha=0.01, gamma=0.9):
        for k in range(n_episodes):
            state = (0, 0)  # Estado inicial
            self.env.state = state  # Configura el estado inicial del entorno
            episode_end = False  # Indica si el episodio ha terminado o no
            epsilon = max(1 / (1 + k), 0.05)  # Calcula el valor de epsilon para la política epsilon-greedy
            
            while not episode_end:
                state = self.env.state  # Obtiene el estado actual
                action_index = self.agent.apply_policy(state, epsilon)  # Aplica la política epsilon-greedy para seleccionar una acción
                action = self.agent.action_space[action_index]  # Obtiene la acción correspondiente al índice seleccionado
                reward, episode_end = self.env.step(action)  # Realiza un paso en el entorno y obtiene la recompensa y el indicador de fin de episodio
                successor_state = self.env.state  # Obtiene el estado sucesor después de realizar la acción
                successor_action_index = self.agent.apply_policy(successor_state, epsilon)  # Aplica la política epsilon-greedy al estado sucesor

                action_value = self.agent.action_function[state[0], state[1], action_index]  # Valor de la acción actual en el estado actual
                successor_action_value = self.agent.action_function[successor_state[0], successor_state[1], successor_action_index]  # Valor de la acción sucesora en el estado sucesor

                q_update = alpha * (reward + gamma * successor_action_value - action_value)  # Cálculo de la actualización Q

                self.agent.action_function[state[0], state[1], action_index] += q_update  # Actualización del valor Q de la acción actual
                self.agent.policy = self.agent.action_function.copy()  # Actualiza la política a partir de la función de acción


    def sarsa_lambda(self, n_episodes=1000, alpha=0.05, gamma=0.9, lamb=0.8):
        for k in range(n_episodes):
            self.agent.E = np.zeros(shape=self.agent.action_function.shape)  # Inicializa la traza de elegibilidad a cero
            state = (0, 0)  # Estado inicial
            self.env.state = state  # Configura el estado inicial del entorno
            episode_end = False  # Indica si el episodio ha terminado o no
            epsilon = max(1 / (1 + k), 0.2)  # Calcula el valor de epsilon para la política epsilon-greedy
            action_index = self.agent.apply_policy(state, epsilon)  # Aplica la política epsilon-greedy para seleccionar una acción
            action = self.agent.action_space[action_index]  # Obtiene la acción correspondiente al índice seleccionado
            
            while not episode_end:
                reward, episode_end = self.env.step(action)  # Realiza un paso en el entorno y obtiene la recompensa y el indicador de fin de episodio
                successor_state = self.env.state  # Obtiene el estado sucesor después de realizar la acción
                successor_action_index = self.agent.apply_policy(successor_state, epsilon)  # Aplica la política epsilon-greedy al estado sucesor

                action_value = self.agent.action_function[state[0], state[1], action_index]  # Valor de la acción actual en el estado actual
                if not episode_end:
                    successor_action_value = self.agent.action_function[successor_state[0], successor_state[1], successor_action_index]  # Valor de la acción sucesora en el estado sucesor
                else:
                    successor_action_value = 0  # Valor 0 si el episodio ha terminado

                delta = reward + gamma * successor_action_value - action_value  # Diferencia temporal de los valores Q
                self.agent.E[state[0], state[1], action_index] += 1  # Incrementa la traza de elegibilidad en la posición correspondiente
                self.agent.action_function = self.agent.action_function + alpha * delta * self.agent.E  # Actualiza la función de acción con la traza de elegibilidad
                self.agent.E = gamma * lamb * self.agent.E  # Decrementa la traza de elegibilidad de acuerdo con el factor de descuento y la tasa de decaimiento lambda
                state = successor_state  # Actualiza el estado actual con el estado sucesor
                action = self.agent.action_space[successor_action_index]  # Actualiza la acción actual con la acción sucesora
                action_index = successor_action_index  # Actualiza el índice de acción actual con el índice de acción sucesora
                self.agent.policy = self.agent.action_function.copy()  # Actualiza la política a partir de la función de acción


    def monte_carlo_learning(self, epsilon=0.1):
        state = (0, 0)  # Estado inicial
        self.env.state = state  # Configura el estado inicial del entorno

        states, actions, rewards = self.play_episode(state, epsilon=epsilon)  # Juega un episodio y obtiene los estados, acciones y recompensas
        
        first_visits = []  # Almacena los estados y acciones visitados por primera vez en el episodio
        for idx, state in enumerate(states):
            action_index = actions[idx]  # Obtiene el índice de la acción correspondiente al estado actual
            if (state, action_index) in first_visits:  # Si ya se ha visitado este estado y acción, continúa al siguiente paso
                continue
            r = np.sum(rewards[idx:])  # Calcula la recompensa total acumulada a partir de este estado y acción en adelante
            if (state, action_index) in self.agent.Returns.keys():  # Si ya hay registros de retornos para este estado y acción
                self.agent.Returns[(state, action_index)].append(r)  # Agrega la recompensa a la lista de retornos
            else:  # Si es la primera vez que se encuentra este estado y acción
                self.agent.Returns[(state, action_index)] = [r]  # Crea una nueva entrada en los retornos
            self.agent.action_function[state[0], state[1], action_index] = np.mean(self.agent.Returns[(state, action_index)])  # Actualiza la función de acción tomando el promedio de los retornos acumulados
            first_visits.append((state, action_index))  # Registra que este estado y acción han sido visitados por primera vez en el episodio

        self.agent.policy = self.agent.action_function.copy()  # Actualiza la política a partir de la función de acción


    def visualize_policy(self):
        greedy_policy = self.agent.policy.argmax(axis=2)  # Obtiene la política greedy seleccionando las acciones con los valores más altos
        policy_visualization = {}  # Diccionario para visualizar la política en forma de flechas o símbolos
        if self.agent.piece == 'king':
            arrows = "↑ ↗ → ↘ ↓ ↙ ← ↖"  # Flechas correspondientes a las acciones del rey
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]  # Fila visual del tablero
        elif self.agent.piece == 'knight':
            arrows = "↑↗ ↗→ →↘ ↓↘ ↙↓ ←↙ ←↖ ↖↑"  # Flechas correspondientes a las acciones del caballo
            visual_row = ["[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]"]  # Fila visual del tablero
        elif self.agent.piece == 'bishop':
            arrows = "↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖"  # Flechas correspondientes a las acciones del alfil
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]  # Fila visual del tablero
        elif self.agent.piece == 'rook':
            arrows = "↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ←"  # Flechas correspondientes a las acciones de la torre
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]  # Fila visual del tablero
        arrowlist = arrows.split(" ")  # Divide las flechas en una lista
        for idx, arrow in enumerate(arrowlist):
            policy_visualization[idx] = arrow  # Asocia cada índice con una flecha en el diccionario de visualización de la política
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())  # Crea el tablero visual a partir de filas visuales copiadas

        for row in range(greedy_policy.shape[0]):
            for col in range(greedy_policy.shape[1]):
                idx = greedy_policy[row, col]  # Obtiene el índice correspondiente a la acción en esta posición del tablero

                visual_board[row][col] = policy_visualization[idx]  # Asigna la flecha correspondiente a la posición del tablero

        visual_board[self.env.terminal_state[0]][self.env.terminal_state[1]] = "F"  # Marca el estado terminal con "F"
        pprint.pprint(visual_board)  # Imprime el tablero visualizado

