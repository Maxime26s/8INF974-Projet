import argparse
import torch
from train import train_dqn, test_dqn
import numpy as np
import random


class HyperParameters:
    def __init__(
        self,
        memory_size,
        batch_size,
        gamma,
        initial_epsilon,
        final_epsilon,
        epsilon_decay,
        learning_rate,
    ):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate


hyperparameters = {
    "CartPole-v1": HyperParameters(
        memory_size=10000,
        batch_size=128,
        gamma=0.99,
        initial_epsilon=1.0,
        final_epsilon=0.01,
        epsilon_decay=1000,
        learning_rate=0.0001,
    ),
    "Acrobot-v1": HyperParameters(
        memory_size=20000,
        batch_size=128,
        gamma=0.99,
        initial_epsilon=1.0,
        final_epsilon=0.01,
        epsilon_decay=2000,
        learning_rate=0.0001,
    ),
    "MountainCar-v0": HyperParameters(
        memory_size=20000,
        batch_size=128,
        gamma=0.99,
        initial_epsilon=1.0,
        final_epsilon=0.01,
        epsilon_decay=2000,
        learning_rate=0.0001,
    ),
}


def main(game, mode, render_mode):
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # game = "CartPole-v1"
    # game = "Acrobot-v1"
    # game = "MountainCar-v0"
    # game = "PongNoFrameskip-v4"

    if mode == "train":
        num_episodes = 600 if torch.cuda.is_available() else 50

        train_dqn(
            game=game,
            render_mode=render_mode,
            num_episodes=num_episodes,
            use_double_dqn=True,
            use_prioritized_memory=True,
        )
    elif mode == "test":
        test_dqn(game=game, model_path="trained_model.pth", num_episodes=100)
    elif mode == "benchmark":
        games = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
        use_double_dqn = [False, True]
        use_prioritized_memory = [False, True]

        for game in games:
            for double_dqn in use_double_dqn:
                for prioritized_memory in use_prioritized_memory:
                    print(
                        f"Training DQN on {game} with {'DDQN' if double_dqn else 'DQN'} and {'Prioritized Memory' if prioritized_memory else 'Regular Memory'}"
                    )
                    train_dqn(
                        game=game,
                        render_mode=render_mode,
                        num_episodes=500 if torch.cuda.is_available() else 50,
                        use_double_dqn=double_dqn,
                        use_prioritized_memory=prioritized_memory,
                        should_plot=False,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test a DQN model.")
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test", "benchmark"],
        help="Mode to run the script in: train, test or benchmark.",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="CartPole-v1",
        help="Game environment to use (e.g., 'PongNoFrameskip-v4' or 'CartPole-v1').",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        choices=[None, "human"],
        help="Render mode for the environment, use 'human' for visual output.",
    )

    args = parser.parse_args()
    main(game=args.game, mode=args.mode, render_mode=args.render_mode)
