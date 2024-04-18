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
        data = data.__array__()
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray) and data.ndim == 3:
        data = np.stack(data, axis=0)

    return torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)


def select_action(state, policy_net, steps_done, n_actions, env):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    # steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
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
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def setup_env(game, render_mode=None):
    env = gym.make(game, render_mode=render_mode)
    state, _ = env.reset(seed=42)
    if len(state.shape) > 1:
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, shape=84)
        env = gym.wrappers.TransformObservation(env, lambda obs: obs / 255.0)
        env = gym.wrappers.FrameStack(env, 4)
    return env


def setup_model(state, n_actions):
    if len(state.shape) > 1:
        policy_net = ImageDQN(state.shape, n_actions).to(device)
        target_net = ImageDQN(state.shape, n_actions).to(device)
    else:
        n_observations = len(state)
        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
    return policy_net, target_net


def perform_action(env, action):
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
        next_state = None
    else:
        next_state = convert_to_tensor(observation, device=device)

    return next_state, reward, done


def update_target_net(policy_net, target_net):
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)


def train_dqn(game, render_mode=None):
    env = setup_env(game, render_mode)
    n_actions = env.action_space.n
    state, _ = env.reset(seed=42)

    policy_net, target_net = setup_model(state, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0
    episode_durations = []

    num_episodes = 600 if torch.cuda.is_available() else 50

    for episode in range(num_episodes):
        state, _ = env.reset(seed=42)
        state = convert_to_tensor(state, device=device)

        for t in count():
            action = select_action(state, policy_net, steps_done, n_actions, env)
            steps_done += 1
            next_state, reward, done = perform_action(env, action)

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer)
            update_target_net(policy_net, target_net)

            if done:
                print(f"Episode: {episode + 1}")

                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break

    env.close()
    torch.save(policy_net.state_dict(), "trained_model.pth")
    print("Complete")

    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()


def test_dqn(model_path, game, num_episodes=10):
    env = setup_env(game, "human")
    n_actions = env.action_space.n
    state, _ = env.reset(seed=42)

    policy_net, _ = setup_model(state, n_actions)
    policy_net.load_state_dict(torch.load(model_path))

    for episode in range(num_episodes):
        state, info = env.reset()
        state = convert_to_tensor(state, device=device)

        total_reward = 0
        for t in count():
            action = select_action_test(state, policy_net)
            state, reward, done = perform_action(env, action)

            total_reward += reward.item()

            if done:
                print(f"Episode: {episode + 1}, Total reward: {total_reward}")
                break
    env.close()
