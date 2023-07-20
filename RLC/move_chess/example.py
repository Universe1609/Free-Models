import numpy as np

class FakeAgent:
    def __init__(self):
        self.action_function = np.random.rand(3,3,4)  # Aquí, simplemente inicializamos con números aleatorios.
        self.policy = np.zeros(shape=self.action_function.shape)
        self.action_space = ['arriba', 'abajo', 'izquierda', 'derecha']  # Acciones posibles

    def apply_policy(self, state, epsilon):
        # Esta es la implementación de apply_policy() que has proporcionado.
        greedy_action_value = np.max(self.policy[state[0], state[1], :])
        greedy_indices = [i for i, a in enumerate(self.policy[state[0], state[1], :]) if a == greedy_action_value]
        action_index = np.random.choice(greedy_indices)
        if np.random.uniform(0, 1) < epsilon:
            action_index = np.random.choice(range(len(self.action_space)))
        return action_index

# Creando el agente
agent = FakeAgent()

# Definimos un estado y un epsilon
state = (1, 2)  # Suponemos que este estado es válido para nuestro espacio de estados 3x3.
epsilon = 0.1

# Usamos apply_policy() en este estado y epsilon.
action_index = agent.apply_policy(state, epsilon)

# Mostramos la acción seleccionada
print(f"La acción seleccionada es: {agent.action_space[action_index]}")