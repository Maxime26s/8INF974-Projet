import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from memory import Transition, ReplayMemory
from prioritized_memory import PrioritizedReplayMemory


class DQNAgent:
    def __init__(
        self,
        policy_net,
        target_net,
        n_actions,
        device,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=1000,
        use_double_dqn=False,
        use_prioritized_memory=False,
        memory_capacity=10000,
        alpha=0.6,  # Priority exponent
        beta=0.4,  # Importance-sampling exponent
    ):
        self.policy_net = policy_net
        self.target_net = target_net
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_double_dqn = use_double_dqn
        self.steps_done = 0
        if use_prioritized_memory:
            self.memory = PrioritizedReplayMemory(memory_capacity, alpha=alpha)
            self.beta = beta
        else:
            self.memory = ReplayMemory(memory_capacity)
        self.optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

        self.loss_history = []

    def get_epsilon(self):
        eps_threshold = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_decay)
        return eps_threshold

    def act(self, state):
        sample = random.random()
        eps_threshold = self.get_epsilon()
        self.steps_done += 1
        if sample > eps_threshold:
            return self.predict(state)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    def predict(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.view(1, 1)

    def remember(self, *args):
        self.memory.push(*args)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        if isinstance(self.memory, PrioritizedReplayMemory):
            transitions, indices, weights = self.memory.sample(
                batch_size, beta=self.beta
            )
            weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        else:
            transitions = self.memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=self.device)

        with torch.no_grad():
            if self.use_double_dqn:
                if non_final_mask.sum() > 0:
                    best_next_actions = (
                        self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    )
                    next_state_values[non_final_mask] = (
                        self.target_net(non_final_next_states)
                        .gather(1, best_next_actions)
                        .squeeze()
                    )
            else:
                next_state_values[non_final_mask] = (
                    self.target_net(non_final_next_states).max(1).values
                )

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        if isinstance(self.memory, PrioritizedReplayMemory):
            criterion = nn.SmoothL1Loss(reduction="none")
            losses = criterion(
                state_action_values, expected_state_action_values.unsqueeze(1)
            ).squeeze()
            weighted_losses = losses * weights
            loss = weighted_losses.mean()
            errors = losses.detach().abs().cpu().numpy()
            self.memory.update_priorities(indices, errors)
        else:
            criterion = nn.SmoothL1Loss()
            loss = criterion(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
