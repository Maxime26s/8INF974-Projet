import argparse
import torch
from train import train_dqn, test_dqn


def main(game, mode, render_mode):
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    # game = "PongNoFrameskip-v4"

    if mode == "train":
        train_dqn(game=game, render_mode=render_mode)
    elif mode == "test":
        test_dqn(game=game, model_path="trained_model.pth", num_episodes=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test a DQN model.")
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test"],
        help="Mode to run the script in: train or test.",
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
