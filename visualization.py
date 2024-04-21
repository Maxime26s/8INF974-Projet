import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("./out", exist_ok=True)

full_name_dict = {
    "cartpole": "CartPole",
    "acrobot": "Acrobot",
    "mountaincar": "MountainCar",
    "ddqn": "Double DQN",
    "dqn": "DQN",
    "prio": "Prioritized Memory",
    "reg": "Regular Memory",
}

rewards = {
    "CartPole": (0, 500),
    "Acrobot": (-500, 0),
    "MountainCar": (-200, 0),
}


def read_and_plot_csv(file_path):
    df = pd.read_csv(file_path)

    plt.figure(figsize=(10, 5))  # Set the size of the figure
    plt.plot(df["episode"], df["reward"], linestyle="-", color="green")
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.xlim(left=0)
    plt.grid(True)
    plt.show()


def group_and_plot_epochs(file_path, n):
    file_name = os.path.basename(file_path)
    parts = file_name.replace(".csv", "").split("_")
    game = full_name_dict.get(parts[0].lower(), parts[0])
    model = full_name_dict.get(parts[1].lower(), parts[1])
    variation = full_name_dict.get(parts[2].lower(), parts[2])
    title = f"{game} - {model} - {variation} - Average Reward per Epoch ({n} episodes per epoch)"

    df = pd.read_csv(file_path)

    df["epoch"] = (df["episode"] - 1) // n
    epoch_data = df.groupby("epoch")["reward"].mean().reset_index()

    epoch_data["epoch"] += 1
    epoch_data["reward"] = epoch_data["reward"] + abs(rewards[game][0])

    start_point = pd.DataFrame({"epoch": [0], "reward": [0]})
    epoch_data = pd.concat([start_point, epoch_data], ignore_index=True)

    plt.figure(figsize=(10, 5))
    plt.plot(
        epoch_data["epoch"],
        epoch_data["reward"],
        linestyle="-",
        color="blue",
    )
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.xlim(left=0, right=epoch_data["epoch"].max())
    plt.ylim(bottom=0, top=rewards[game][1] - rewards[game][0])
    plt.grid(True)
    plt.savefig(f"./out/{game}_{model}_{variation}.png")


result_folder = "./results"

files = [
    f
    for f in os.listdir(result_folder)
    if os.path.isfile(os.path.join(result_folder, f))
]
print(files)

for file in files:
    group_and_plot_epochs(os.path.join(result_folder, file), 10)
