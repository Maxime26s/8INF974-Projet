import torch
import torch.optim as optim
import torch.nn as nn
from dqn_model import DQN
from agent import DQNAgent
from memory import ReplayMemory, Transition
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
    elif isinstance(data, np.ndarray):
        # If the data is already a 3D image, just add the batch dimension
        if data.ndim == 3:  # For image data [C, H, W]
            data = data[None, :]  # Add batch dimension if needed
        elif data.ndim == 1:  # For flattened arrays or single-dimensional states
            data = data.reshape(1, -1)  # Reshape to [1, feature_length]

    # Do not add an extra unsqueeze unless required
    return torch.tensor(data, dtype=torch.float32, device=device)


def setup_env(game, num_envs=4, render_mode=None):
    envs = gym.make_vec(
        game, num_envs=num_envs, render_mode=render_mode, vectorization_mode="sync"
    )
    observation_shape = envs.single_observation_space.shape

    if len(observation_shape) > 2:
        envs = gym.wrappers.VectorListWrapper(envs, gym.wrappers.GrayScaleObservation)
        envs = gym.wrappers.VectorListWrapper(
            envs, gym.wrappers.ResizeObservation, shape=84
        )
        envs = gym.wrappers.VectorListWrapper(
            envs, gym.wrappers.TransformObservation, lambda obs: obs / 255.0
        )
        envs = gym.wrappers.VectorListWrapper(
            envs, gym.wrappers.FrameStack, num_stack=4
        )

    return envs


def setup_model(observation_shape, n_actions):
    policy_net = DQN(observation_shape, n_actions).to(device)
    target_net = DQN(observation_shape, n_actions).to(device)
    return policy_net, target_net


def perform_action(envs, actions):
    actions = actions.cpu().numpy().flatten()

    observations, rewards, terminated, truncated, _ = envs.step(actions)
    next_observations = convert_to_tensor(observations, device=device)
    rewards = torch.tensor(rewards, device=device).unsqueeze(1)
    done = torch.tensor(terminated | truncated, dtype=torch.bool, device=device)

    for i in range(len(done)):
        if done[i]:
            next_observations[i] = torch.zeros_like(next_observations[i])

    return next_observations, rewards, done


def train_dqn(game, num_envs=2, render_mode=None):
    envs = setup_env(game, num_envs, render_mode)
    observation_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    policy_net, target_net = setup_model(observation_shape, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    agent = DQNAgent(policy_net, target_net, n_actions, device)

    episode_durations = []

    num_episodes = 600 if torch.cuda.is_available() else 50

    for episode in range(num_episodes):
        observations, _ = envs.reset(seed=42)
        observations = convert_to_tensor(observations, device=device)

        for frame in count():
            actions = agent.act(observations)
            next_observations, rewards, done = perform_action(envs, actions)

            for i in range(envs.num_envs):
                agent.remember(
                    observations[i], actions[i], next_observations[i], rewards[i]
                )

            observations = next_observations

            agent.replay(BATCH_SIZE)
            agent.update_target()

            if done.all():
                print(f"Episode: {episode + 1}")

                episode_durations.append(frame + 1)
                plot_durations(episode_durations)
                break

    envs.close()
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
