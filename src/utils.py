import matplotlib
import matplotlib.pyplot as plt
import torch
import os
import csv

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def plot_metrics(
    episode_durations, episode_rewards, loss_per_episode, show_result=False
):
    plt.figure(1, figsize=(12, 8))  # Adjust the figure size for three subplots
    plt.clf()

    # Plot durations
    plt.subplot(2, 2, 1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title("Episode Durations")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.xlim(left=0)
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Plot rewards
    plt.subplot(2, 2, 2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.xlim(left=0)
    plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Plot losses
    plt.subplot(2, 2, 3)
    losses_t = torch.tensor(loss_per_episode, dtype=torch.float)
    plt.title("Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.xlim(left=0)
    plt.plot(losses_t.numpy())
    if len(losses_t) >= 100:
        means = losses_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.clear_output(wait=True)
        display.display(plt.gcf())


def initialize_metrics_csv(session_dir):
    metrics_path = os.path.join(session_dir, "metrics_history.csv")
    with open(metrics_path, "w", newline="") as csvfile:
        fieldnames = ["episode", "duration", "reward", "average_loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    return metrics_path


def append_metrics_to_csv(metrics_path, episode, duration, reward, average_loss):
    with open(metrics_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["episode", "duration", "reward", "average_loss"]
        )
        writer.writerow(
            {
                "episode": episode,
                "duration": duration,
                "reward": reward,
                "average_loss": average_loss,
            }
        )
