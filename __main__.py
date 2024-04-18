import torch
from train import train_dqn, test_dqn

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

if __name__ == "__main__":
    game = "PongNoFrameskip-v4"
    game = "CartPole-v1"

    render_mode = None
    render_mode = "human"

    train_dqn(game=game, render_mode=render_mode)
    # test_dqn(game, "trained_model.pth", num_episodes=100)
