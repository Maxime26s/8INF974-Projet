import torch
import torch.optim as optim
import torch.nn as nn
from dqn_model import DQN, ImageDQN, ReplayMemory, Transition
from utils import plot_durations
import math
import gymnasium as gym
import random
from itertools import count
import matplotlib.pyplot as plt
import numpy as np

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_to_tensor(data, device):
    if isinstance(data, gym.wrappers.LazyFrames):
        data = data.__array__()  # Convert LazyFrames to numpy array
    elif isinstance(data, list):
        data = np.array(data)  # Convert list of numpy arrays to a single numpy array
    elif isinstance(data, np.ndarray) and data.ndim == 3:
        data = np.stack(data, axis=0)  # Stack along the first axis if needed

    # Convert numpy array to torch tensor
    return torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)


def select_action(state, policy_net, steps_done, n_actions, env):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    # steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


def select_action_test(state, policy_net):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train_dqn(game, render_mode=None):
    # env = gym.make(game, render_mode=render_mode)
    env = gym.make("PongNoFrameskip-v4", render_mode=render_mode)
    env = gym.wrappers.GrayScaleObservation(env)  # Convert observation to grayscale
    env = gym.wrappers.ResizeObservation(env, shape=84)  # Resize frame to 84x84
    env = gym.wrappers.TransformObservation(
        env, lambda obs: obs / 255.0
    )  # Normalize pixel values
    env = gym.wrappers.FrameStack(env, 4)  # Stack 4 frames together

    n_actions = env.action_space.n
    state, info = env.reset(seed=42)
    n_observations = len(state)
    print(state.shape)

    if len(state.shape) > 1:
        policy_net = ImageDQN(state.shape, n_actions).to(device)
        target_net = ImageDQN(state.shape, n_actions).to(device)
    else:
        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0
    episode_durations = []

    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset(seed=42)

        # Now, convert the numpy array to a torch tensor
        state = convert_to_tensor(state, device=device)

        for t in count():
            action = select_action(state, policy_net, steps_done, n_actions, env)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = convert_to_tensor(observation, device=device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, policy_net, target_net, optimizer)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                print(f"Episode: {i_episode + 1}")

                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break

    torch.save(policy_net.state_dict(), "trained_model.pth")
    env.close()
    print("Complete")
    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()


def test_dqn(model_path, game, num_episodes=10):
    env = gym.make(game, render_mode="human")
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path))

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        for t in count():
            action = select_action_test(state, policy_net)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            total_reward += reward
            if terminated or truncated:
                print(f"Episode: {i_episode + 1}, Total reward: {total_reward}")
                break
    env.close()
