import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm


class NatureEncoder(nn.Module):
    def __init__(self, input_shape: tuple, args: dict):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[1], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        if "weights" in args:
            self.cnn.load_state_dict(torch.load(args["weights"]))

        self.input_shape = input_shape

    def forward(self, x):
        if x.max() > 1.0:
            x = x / 255.0

        return self.cnn(x)

    def get_flat_feats(self):
        x = torch.rand(size=self.input_shape)
        with torch.no_grad():
            x = self.cnn(x)

        return x.numel()

class EncoderVAE(nn.Sequential):
    def __init__(self, input_shape, custom_encoder_class=None, custom_encoder_args=None, latent_size=32):
        super().__init__()
        if custom_encoder_class is not None:
            assert "train_cnn" in custom_encoder_args.keys(), "You must specify whether to train the CNN"
        self.input_shape = input_shape
        self.latent_size = latent_size
        if custom_encoder_class is not None:
            self.cnn = custom_encoder_class(
                input_shape=input_shape,
                args=custom_encoder_args,
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=input_shape[1], out_channels=32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
                nn.ReLU()
            )

        n_flat = self.get_flat_feats(self.input_shape)

        self.mu = nn.Linear(in_features=n_flat, out_features=latent_size)
        self.logvar = nn.Linear(in_features=n_flat, out_features=latent_size)

        self._train_cnn = custom_encoder_args["train_cnn"] if custom_encoder_class is not None else True
        # TODO: choose in main script
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        if self._train_cnn:
            z = self.cnn(x)
            z_flat = torch.flatten(z, start_dim=1)
        else:
            with torch.no_grad():
                z_flat = self.cnn(x)

        mu_hat = self.mu(z_flat)
        logvar_hat = self.logvar(z_flat)

        # Re-parametrization
        z = self._reparameterize(mu_hat, logvar_hat)

        # We need mu and logvar for loss computation
        return z, mu_hat, logvar_hat

    def _reparameterize(self, mu, logvar):
        """dist = Normal(mu, torch.exp(logvar))

        return dist.rsample()"""
        # TODO: change
        noise = torch.normal(torch.zeros(size=mu.size()), torch.ones(size=mu.size())).to(self.device)
        return mu + torch.exp(logvar) * noise

    def get_conv_output_shape(self, input_shape):
        x = torch.rand(input_shape)
        with torch.no_grad():
            x = self.cnn(x)

        return x.size()

    def get_flat_feats(self, input_shape):
        x = torch.rand(input_shape)
        with torch.no_grad():
            x = self.cnn(x)

        return x.size().numel()

    def get_latent_size(self):
        return self.latent_size


class DecoderVAE(nn.Sequential):
    def __init__(self, output_shape, mlp_out_feats, enc_conv_out_shape, latent_size=32):
        super().__init__()
        self.output_shape = output_shape
        self.enc_conv_output_shape = enc_conv_out_shape
        self.latent_size = latent_size

        self.mlp = nn.Linear(in_features=latent_size, out_features=mlp_out_feats)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=mlp_out_feats, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=output_shape[1], kernel_size=6, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.mlp(x)
        # Hardcoded for SuperTuxKart...
        z = self.deconv(z.view(z.size(0), 1024, 1, 1))
        #z = z.view(x.size(0), *self.enc_conv_output_shape[1:])
        # TODO: check size
        #z = self.deconv(z)

        return z


class VAE(nn.Sequential):
    def __init__(self, custom_encoder_class=None, custom_encoder_args=None, input_shape=(1, 3, 64, 64), latent_size=32, weights_path=""):
        super().__init__()
        self.encoder = EncoderVAE(input_shape, custom_encoder_class, custom_encoder_args, latent_size)
        n_flat_feats = self.encoder.get_flat_feats(input_shape)
        enc_conv_out_shape = self.encoder.get_conv_output_shape(input_shape)
        self.decoder = DecoderVAE(input_shape, latent_size=latent_size, mlp_out_feats=n_flat_feats, enc_conv_out_shape=enc_conv_out_shape)
        if weights_path:
            print(f"Loading {weights_path.split(sep='/')[-1]}...")
            self.load_state_dict(torch.load(weights_path))
            previous_model_name = weights_path.split(sep="/")[-1]
            self.num_previous_epochs = int((previous_model_name.split(sep=".")[0]).split(sep="_")[2])
        else:
            self.num_previous_epochs = 0

        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, mu, logvar

    def get_latent(self, x):
        with torch.no_grad():
            z, mu, logvar = self.encoder(x)

        return z, mu, logvar

    def decode(self, z):
        x_hat = self.decoder(z)

        return x_hat

    def compute_loss(self, x, x_hat, mu, logvar, kl_multiplier=0.05):
        mse = self.mse_loss(x_hat, x)
        #kl_divergence = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
        # Taken from https://arxiv.org/pdf/1312.6114 (Appendix B)
        kl_divergence = -0.5 * torch.sum(1 + 2 * logvar - torch.pow(mu, 2) - torch.exp(2 * logvar))

        return mse + kl_multiplier * kl_divergence, mse, kl_multiplier * kl_divergence

    def train_model(self, dataset, test_dataset, batch_size, epochs, lr, device, save_path="", model_name="", game="", plot_freq=20, start_temp=0.0, end_temp=0.001, temp_epochs=None,
                    loss_type="mse_kl", run_name="", track=False, use_history=False):
        self.to(device)

        data_loader = DataLoader(dataset, batch_size, shuffle=True)
        num_batches = len(data_loader)
        optimizer = Adam(self.parameters(), lr=lr)

        if track:
            wandb.init(project="wm-vae-training", sync_tensorboard=True, name=run_name)

        progress_bar = tqdm(range(self.num_previous_epochs, epochs + self.num_previous_epochs, 1))

        kl_temperature = 0.0
        if temp_epochs is None:
            temp_epochs = [0, epochs]

        temp_update = (end_temp - start_temp) / (temp_epochs[1] - temp_epochs[0])

        writer = SummaryWriter(log_dir='vae_logs/gaussian/')

        for e in progress_bar:
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_mse = 0.0
            if temp_epochs[0] <= e <= temp_epochs[1]:
                kl_temperature += temp_update

            for i, batch in enumerate(data_loader):
                batch = batch.to(device)

                x_hat, mu, logvar = self.forward(batch)

                loss, mse, kl = self.compute_loss(batch, x_hat, mu, logvar, kl_multiplier=kl_temperature)

                optimizer.zero_grad()
                if torch.isnan(loss):
                    print("Hello")
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_kl += kl
                epoch_mse += mse
                avg_loss = epoch_loss / (i + 1)

                if loss_type == "ssim":
                    progress_bar.set_description(f"[{game} - BATCH {i}/{num_batches}] Avg. Loss: {avg_loss}")
                else:
                    progress_bar.set_description(f"[{game} - BATCH {i}/{num_batches}] Avg. Loss: {avg_loss} (MSE: {mse} - KL: {kl}) | KL temp: {kl_temperature}")

            # DEBUG: get some insight on a per-epoch basis
            if test_dataset is not None and e % plot_freq == 0:
                self.test(dataset=test_dataset, device=device, latent_size=self.encoder.latent_size)

            writer.add_scalar("Avg. Epoch Loss", epoch_loss / num_batches, e)
            writer.add_scalar("KL", epoch_kl / num_batches, e)
            writer.add_scalar("MSE", epoch_mse / num_batches, e)

        if save_path != "":
            os.makedirs(save_path, exist_ok=True)
            if model_name == "":
                model_name = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
            torch.save(self.state_dict(), f"{save_path}/{model_name}.pth")

    def reconstruct(self, x):
        with torch.no_grad():
            return self.forward(x)

    def test(self, dataset, device, latent_size, sample_size=10):
        samples = np.random.randint(0, len(dataset), size=sample_size)

        fig, ax = plt.subplots(nrows=2, ncols=sample_size, figsize=(15, 3))

        for i, idx in enumerate(samples):
            img = dataset.__getitem__(idx)
            rec, _, _ = self.reconstruct(img.unsqueeze(0).to(device))

            ax[0, i].imshow(img.permute(1, 2, 0).numpy())
            ax[1, i].imshow(((rec.squeeze(0)).permute(1, 2, 0)).cpu().numpy())
            ax[0, i].axis('off')
            ax[1, i].axis('off')

        fig.suptitle(f"{dataset.prefix_path.split(sep='/')[-2]} - Latent space size {latent_size}")

        plt.tight_layout()
        plt.show()
