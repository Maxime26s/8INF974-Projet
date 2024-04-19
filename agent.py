import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from memory import Transition, ReplayMemory

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000


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
        epsilon=0.9,
    ):
        self.policy_net = policy_net
        self.target_net = target_net
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.steps_done = 0
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

    def act(self, observations):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            return self.predict(observations)
        else:
            return torch.tensor(
                [
                    [random.randrange(self.n_actions)]
                    for _ in range(observations.size(0))
                ],
                device=self.device,
                dtype=torch.long,
            )

    def predict(self, states):
        with torch.no_grad():
            return self.policy_net(states).max(1).indices.view(-1, 1)

    def remember(self, *args):
        self.memory.push(*args)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        print("next_state_batch shape:", reward_batch.shape)

        non_final_mask = next_state_batch.sum() != 0
        non_final_next_states = next_state_batch[non_final_mask]

        print(f"Intended non-final states count: {non_final_mask.sum().item()}")
        print(f"Actual non-final next states count: {non_final_next_states.size(0)}")

        if non_final_next_states.size(0) != non_final_mask.sum():
            raise Exception("Non-final states count mismatch!")

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=self.device)

        print(f"Batch size: {batch_size}")
        print(f"next_state_values shape: {next_state_values.shape}")
        print(f"non_final_mask shape (True count): {non_final_mask.sum()}")
        print(
            f"Shape of target network output: {self.target_net(non_final_next_states).max(1).values.shape}"
        )

        print(
            f"Non-final next states shape: {non_final_next_states.shape}"
        )  # Debugging line

        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                temp_values = self.target_net(non_final_next_states).max(1).values
                print(f"Temp values shape: {temp_values.shape}")  # Debugging line
                if temp_values.shape[0] != non_final_mask.sum():
                    print("Mismatch detected")
                    raise ValueError("Shape mismatch in DQN target calculation.")
                next_state_values[non_final_mask] = temp_values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def update_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
