import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("./out", exist_ok=True)
os.makedirs("./out/data", exist_ok=True)

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


def save_epoch_data_to_csv(epoch_data, game, model, variation):
    csv_filename = f"./out/data/{game}_{model}_{variation}_epoch_data.csv".replace(
        " ", "_"
    )
    epoch_data.to_csv(csv_filename, index=False)
    print(f"Saved epoch data to {csv_filename}")


def process_file(file_path, episode_per_epoch):
    file_name = os.path.basename(file_path)
    parts = file_name.replace(".csv", "").split("_")
    game = full_name_dict.get(parts[0].lower(), parts[0])
    model = full_name_dict.get(parts[1].lower(), parts[1])
    variation = full_name_dict.get(parts[2].lower(), parts[2])
    title = f"{game} - {model} - {variation} - Average Reward per Epoch ({episode_per_epoch} episodes per epoch)"

    df = pd.read_csv(file_path)
    df["epoch"] = (df["episode"] - 1) // episode_per_epoch
    epoch_data = df.groupby("epoch")["reward"].mean().reset_index()
    epoch_data["epoch"] += 1
    epoch_data["reward"] = epoch_data["reward"] + abs(rewards[game][0])

    start_point = pd.DataFrame({"epoch": [0], "reward": [0]})
    epoch_data = pd.concat([start_point, epoch_data], ignore_index=True)

    # save_epoch_data_to_csv(epoch_data, game, model, variation)

    return epoch_data, game, model, variation, title


def performance_analysis(epoch_data, game, title):
    if game == "MountainCar":
        good_score_threshold = 50
    elif game == "CartPole":
        good_score_threshold = 100
    elif game == "Acrobot":
        good_score_threshold = 350
    good_epochs = epoch_data[epoch_data["reward"] >= good_score_threshold]
    if not good_epochs.empty:
        first_good_epoch = good_epochs["epoch"].iloc[0]
    else:
        first_good_epoch = None

    max_score = epoch_data["reward"].max()
    max_score_epoch = epoch_data.loc[epoch_data["reward"].idxmax(), "epoch"]

    print(f"{title}")
    print(f"First good scoring epoch: {first_good_epoch}")
    print(f"Max score: {max_score} at epoch {max_score_epoch}")


def plot_raw_episodes(file_path):
    df = pd.read_csv(file_path)

    plt.figure()
    plt.plot(df["episode"], df["reward"])
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.xlim(left=0)
    plt.grid(True)
    plt.show()


def plot_epochs(file_path, episode_per_epoch):
    epoch_data, game, model, variation, title = process_file(
        file_path, episode_per_epoch
    )

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

    performance_analysis(epoch_data, game, title)


def plot_all_experiments(files, n):
    game_data = {}

    for file in files:
        epoch_data, game, model, variation, title = process_file(
            os.path.join(result_folder, file), n
        )
        if game not in game_data:
            game_data[game] = []
        game_data[game].append((epoch_data, model, variation, title))

    for game, data_list in game_data.items():
        plt.figure(figsize=(10, 5))
        for epoch_data, model, variation, title in data_list:
            plt.plot(
                epoch_data["epoch"],
                epoch_data["reward"],
                label=f"{model} - {variation}",
            )
        plt.title(f"{game} - Average Reward per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.xlim(left=0)
        plt.ylim(bottom=0, top=rewards[game][1] - rewards[game][0])
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./out/{game}_summary.png")
        plt.show()


result_folder = "./results"

files = []
for f in os.listdir(result_folder):
    if os.path.isfile(os.path.join(result_folder, f)):
        files.append(f)

plot_all_experiments(files, 10)
