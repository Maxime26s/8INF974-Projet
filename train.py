import torch
import torch.optim as optim
import torch.nn as nn
from dqn_model import DQN, ImageDQN
from agent import DQNAgent
from replay import ReplayMemory, Transition
from utils import plot_durations
import math
import gymnasium as gym
import random
from itertools import count
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_to_tensor(data, device):
    if isinstance(data, gym.wrappers.LazyFrames):
        data = data.__array__()
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray) and data.ndim == 3:
        data = np.stack(data, axis=0)

    return torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)


def setup_env(game, render_mode=None):
    env = gym.make(game, render_mode=render_mode)
    state, _ = env.reset(seed=42)

    if len(state.shape) > 1:
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, shape=84)
        env = gym.wrappers.TransformObservation(env, lambda obs: obs / 255.0)
        env = gym.wrappers.FrameStack(env, 4)

    return env


def setup_model(state_shape, n_actions):
    if len(state_shape) > 1:
        policy_net = ImageDQN(state_shape, n_actions).to(device)
        target_net = ImageDQN(state_shape, n_actions).to(device)
    else:
        policy_net = DQN(state_shape[0], n_actions).to(device)
        target_net = DQN(state_shape[0], n_actions).to(device)
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


def train_dqn(game, render_mode=None):
    env = setup_env(game, render_mode)
    n_actions = env.action_space.n
    state, _ = env.reset(seed=42)

    policy_net, target_net = setup_model(state.shape, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    agent = DQNAgent(policy_net, target_net, n_actions, device)

    episode_durations = []

    num_episodes = 600 if torch.cuda.is_available() else 50

    for episode in range(num_episodes):
        state, _ = env.reset(seed=42)
        state = convert_to_tensor(state, device=device)

        for frame in count():
            action = agent.act(state)
            next_state, reward, done = perform_action(env, action)

            agent.remember(state, action, next_state, reward)

            state = next_state

            agent.replay(BATCH_SIZE)
            agent.update_target()

            if done:
                print(f"Episode: {episode + 1}")

                episode_durations.append(frame + 1)
                plot_durations(episode_durations)
                break

    env.close()
    torch.save(policy_net.state_dict(), "trained_model.pth")
    print("Training completed. Model saved as trained_model.pth")

    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()


def test_dqn(game, model_path, num_episodes=10):
    env = setup_env(game, "human")
    n_actions = env.action_space.n
    state, _ = env.reset(seed=42)

    policy_net, _ = setup_model(state.shape, n_actions)
    policy_net.load_state_dict(torch.load(model_path))

    agent = DQNAgent(policy_net, policy_net, n_actions, device)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = convert_to_tensor(state, device=device)

        total_reward = 0
        for frame in count():
            action = agent.predict(state)
            state, reward, done = perform_action(env, action)

            total_reward += reward.item()

            if done:
                print(f"Episode: {episode + 1}, Total reward: {total_reward}")
                break

    env.close()

    print("Testing completed.")
