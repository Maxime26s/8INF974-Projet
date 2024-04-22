import torch
from dqn_model import DQN
from agent import DQNAgent
from utils import plot_metrics, initialize_metrics_csv, append_metrics_to_csv
import gymnasium as gym
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

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
    obseration_shape = env.observation_space.shape

    if len(obseration_shape) > 1:
        print("Preprocessing for image-based observation space")
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, shape=84)
        env = gym.wrappers.TransformObservation(env, lambda obs: obs / 255.0)
        env = gym.wrappers.FrameStack(env, 4)
    else:
        print("Preprocessing for vector-based observation space")

    return env


def setup_model(observation_shape, n_actions):
    policy_net = DQN(observation_shape, n_actions).to(device)
    target_net = DQN(observation_shape, n_actions).to(device)
    return policy_net, target_net


def perform_action(env, action):
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
        next_observation = None
    else:
        next_observation = convert_to_tensor(observation, device=device)

    return next_observation, reward, done


def train_dqn(
    game,
    render_mode=None,
    num_episodes=100,
    use_double_dqn=False,
    use_prioritized_memory=False,
    save_interval=100,
    should_plot=True,
):
    env = setup_env(game, render_mode)
    n_actions = env.action_space.n
    observation_shape = env.observation_space.shape

    policy_net, target_net = setup_model(observation_shape, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    agent = DQNAgent(
        policy_net,
        target_net,
        n_actions,
        device,
        use_double_dqn=use_double_dqn,
        use_prioritized_memory=use_prioritized_memory,
    )

    episode_durations = []
    episode_rewards = []
    episode_average_loss = []

    model_type = "DDQN" if use_double_dqn else "DQN"
    memory_type = "PrioritizedMemory" if use_prioritized_memory else "StandardMemory"
    session_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    session_dir = f"./{game}/{model_type}/{memory_type}/{session_timestamp}"
    os.makedirs(session_dir, exist_ok=True)

    metrics_path = initialize_metrics_csv(session_dir)

    for episode in range(num_episodes):
        observation, _ = env.reset(seed=42)
        observation = convert_to_tensor(observation, device=device)

        total_reward = 0
        losses = []

        for frame in count():
            action = agent.act(observation)
            next_observation, reward, done = perform_action(env, action)
            total_reward += reward.item()

            agent.remember(observation, action, next_observation, reward)

            observation = next_observation

            loss = agent.replay(BATCH_SIZE)
            if loss is not None:
                losses.append(loss)

            agent.update_target()

            if done:
                episode_durations.append(frame + 1)
                episode_rewards.append(total_reward)
                episode_average_loss.append(np.mean(losses) if losses else 0)

                print(
                    f"Episode: {episode + 1}, Total reward: {total_reward}, Average Loss: {episode_average_loss[-1]}"
                )

                append_metrics_to_csv(
                    metrics_path,
                    episode + 1,
                    episode_durations[-1],
                    total_reward,
                    episode_average_loss[-1],
                )

                if should_plot:
                    plot_metrics(
                        episode_durations, episode_rewards, episode_average_loss
                    )
                break

        if (episode + 1) % save_interval == 0:
            model_path = f"{session_dir}/{episode+1}.pth"
            torch.save(policy_net.state_dict(), model_path)
            print(f"Model saved at {model_path}")

    env.close()
    print("Training completed.")
    final_model_path = f"{session_dir}/final_model.pth"
    torch.save(policy_net.state_dict(), final_model_path)
    print(f"Final model saved as {final_model_path}")

    if should_plot:
        plot_metrics(
            episode_durations, episode_rewards, episode_average_loss, show_result=True
        )
        plt.ioff()
        plt.show()


def test_dqn(game, model_path, num_episodes=10):
    env = setup_env(game, "human")
    n_actions = env.action_space.n
    observation = env.observation_space.shape

    policy_net, _ = setup_model(observation.shape, n_actions)
    policy_net.load_state_dict(torch.load(model_path))

    agent = DQNAgent(policy_net, policy_net, n_actions, device)

    with torch.no_grad():
        for episode in range(num_episodes):
            observation, _ = env.reset()
            observation = convert_to_tensor(observation, device=device)

            total_reward = 0
            for frame in count():
                action = agent.predict(observation)
                observation, reward, done = perform_action(env, action)

                total_reward += reward.item()

                if done:
                    print(f"Episode: {episode + 1}, Total reward: {total_reward}")
                    break

    env.close()

    print("Testing completed.")
