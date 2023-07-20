import numpy as np
import pprint


class Reinforce(object):

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def play_episode(self, state, max_steps=1e3, epsilon=0.1):

        self.env.state = state
        states = []
        actions = []
        rewards = []
        episode_end = False

        count_steps = 0
        while not episode_end:
            count_steps += 1
            states.append(state)
            action_index = self.agent.apply_policy(state, epsilon)
            action = self.agent.action_space[action_index]
            actions.append(action_index)
            reward, episode_end = self.env.step(action)
            state = self.env.state
            rewards.append(reward)
            
            if count_steps > max_steps:
                episode_end = True

        return states, actions, rewards

    def sarsa_td(self, n_episodes=1000, alpha=0.01, gamma=0.9):
        for k in range(n_episodes):
            state = (0, 0)
            self.env.state = state
            episode_end = False
            epsilon = max(1 / (1 + k), 0.05)
            while not episode_end:
                state = self.env.state
                action_index = self.agent.apply_policy(state, epsilon)
                action = self.agent.action_space[action_index]
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.agent.apply_policy(successor_state, epsilon)

                action_value = self.agent.action_function[state[0], state[1], action_index]
                successor_action_value = self.agent.action_function[successor_state[0],
                                                                    successor_state[1], successor_action_index]

                q_update = alpha * (reward + gamma * successor_action_value - action_value)

                self.agent.action_function[state[0], state[1], action_index] += q_update
                self.agent.policy = self.agent.action_function.copy()

    def sarsa_lambda(self, n_episodes=1000, alpha=0.05, gamma=0.9, lamb=0.8):
        for k in range(n_episodes):
            self.agent.E = np.zeros(shape=self.agent.action_function.shape)
            state = (0, 0)
            self.env.state = state
            episode_end = False
            epsilon = max(1 / (1 + k), 0.2)
            action_index = self.agent.apply_policy(state, epsilon)
            action = self.agent.action_space[action_index]
            while not episode_end:
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.agent.apply_policy(successor_state, epsilon)

                action_value = self.agent.action_function[state[0], state[1], action_index]
                if not episode_end:
                    successor_action_value = self.agent.action_function[successor_state[0],
                                                                        successor_state[1], successor_action_index]
                else:
                    successor_action_value = 0
                delta = reward + gamma * successor_action_value - action_value
                self.agent.E[state[0], state[1], action_index] += 1
                self.agent.action_function = self.agent.action_function + alpha * delta * self.agent.E
                self.agent.E = gamma * lamb * self.agent.E
                state = successor_state
                action = self.agent.action_space[successor_action_index]
                action_index = successor_action_index
                self.agent.policy = self.agent.action_function.copy()

    def monte_carlo_learning(self, epsilon=0.1):
        state = (0, 0)
        self.env.state = state

        states, actions, rewards = self.play_episode(state, epsilon=epsilon)

        first_visits = []
        for idx, state in enumerate(states):
            action_index = actions[idx]
            if (state, action_index) in first_visits:
                continue
            r = np.sum(rewards[idx:])
            if (state, action_index) in self.agent.Returns.keys():
                self.agent.Returns[(state, action_index)].append(r)
            else:
                self.agent.Returns[(state, action_index)] = [r]
            self.agent.action_function[state[0], state[1], action_index] = \
                np.mean(self.agent.Returns[(state, action_index)])
            first_visits.append((state, action_index))
        self.agent.policy = self.agent.action_function.copy()

    def visualize_policy(self):
        greedy_policy = self.agent.policy.argmax(axis=2)
        policy_visualization = {}
        if self.agent.piece == 'king':
            arrows = "↑ ↗ → ↘ ↓ ↙ ← ↖"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        elif self.agent.piece == 'knight':
            arrows = "↑↗ ↗→ →↘ ↓↘ ↙↓ ←↙ ←↖ ↖↑"
            visual_row = ["[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]"]
        elif self.agent.piece == 'bishop':
            arrows = "↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        elif self.agent.piece == 'rook':
            arrows = "↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ←"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        arrowlist = arrows.split(" ")
        for idx, arrow in enumerate(arrowlist):
            policy_visualization[idx] = arrow
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())

        for row in range(greedy_policy.shape[0]):
            for col in range(greedy_policy.shape[1]):
                idx = greedy_policy[row, col]

                visual_board[row][col] = policy_visualization[idx]

        visual_board[self.env.terminal_state[0]][self.env.terminal_state[1]] = "F"
        pprint.pprint(visual_board)
