from collections import namedtuple
import numpy as np

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6):
        self.alpha = (
            alpha  # Controls the amount of prioritization (0 means no prioritization)
        )
        self.capacity = capacity
        self.memory = []
        self.priorities = []  # Parallel array of priorities
        self.position = 0

    def push(self, state, action, next_state, reward):
        """Save a transition with priority"""
        max_priority = (
            max(self.priorities) if self.memory else 1.0
        )  # Start with the highest or default priority

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, action, next_state, reward))
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = Transition(state, action, next_state, reward)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of transitions, weighted by priority"""
        total = len(self.memory)
        weights = np.array(self.priorities) ** self.alpha
        p_sum = weights.sum()
        probabilities = weights / p_sum
        indices = np.random.choice(range(total), batch_size, p=probabilities)
        samples = [self.memory[i] for i in indices]

        # Calculate importance-sampling weights
        max_weight = (p_sum * probabilities.min()) ** (-beta)
        weights = (p_sum * probabilities[indices]) ** (-beta)
        normalized_weights = weights / max_weight

        return samples, indices, normalized_weights

    def update_priorities(self, indices, errors):
        """Update priorities of sampled transitions"""
        for index, error in zip(indices, errors):
            self.priorities[index] = error + 1e-5  # Avoid zero priority

    def __len__(self):
        return len(self.memory)
