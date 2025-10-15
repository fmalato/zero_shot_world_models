import os

import torch

from vae import VAE
from data import StoredDataset, StoredDatasetFromVideo


if __name__ == '__main__':
    input_shape = (1, 3, 64, 64)
    latent_size = 512
    num_epochs = 50
    learning_rate = 3e-4
    batch_size = 128

    data_path = "minerl/"
    task_name = "MineRLNavigate-v0"
    trajectories_path = os.path.join(data_path, task_name)
    trajectories = os.listdir(trajectories_path)
    train_test_split = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(input_shape=input_shape, latent_size=latent_size)
    model.to(device)
    model.train()

    dataset = StoredDatasetFromVideo(
        prefix_path=trajectories_path,
        file_paths=trajectories[:int(len(trajectories) * train_test_split)],
    )
    test_dataset = StoredDatasetFromVideo(
        prefix_path=trajectories_path,
        file_paths=trajectories[int(len(trajectories) * train_test_split):],
    )
    model.train_model(
        dataset=dataset,
        test_dataset=test_dataset,
        epochs=num_epochs,
        lr=learning_rate,
        device=device,
        save_path=f"vae_models/{task_name}",
        model_name=f"encoder_{latent_size}_{num_epochs}.pth",
        game=task_name,
        batch_size=batch_size,
        plot_freq=int(num_epochs * 0.1),
        start_temp=0,
        end_temp=1e-8,
        temp_epochs=[0.1 * num_epochs, 0.75 * num_epochs]
    )