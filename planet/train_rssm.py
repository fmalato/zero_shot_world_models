import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt

from rssm_model import RecurrentStateSpaceModel, VisualDecoder, VisualEncoder
from memory import Memory
from main import generate_fictitious_episode
from utils_planet import preprocess_img, bottle


if __name__ == '__main__':
    action_size = 9
    embedding_size = 128
    state_size = 32
    hidden_size = 200
    epochs = 100
    iterations_per_epoch = 200
    batch_size = 64
    horizon = 20
    train_data = "vae_data/lighthouse"
    beta = 1.0
    test_frequency = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = VisualEncoder(
        embedding_size=embedding_size
    )
    decoder = VisualDecoder(
        state_size=state_size,
        latent_size=embedding_size,
        embedding_size=embedding_size
    )
    rssm = RecurrentStateSpaceModel(
        action_size=action_size,
        state_size=state_size,
        latent_size=embedding_size,
        hidden_size=hidden_size,
        embed_size=embedding_size
    )

    encoder.to(device)
    decoder.to(device)
    rssm.to(device)

    buffer = Memory(size=len(os.listdir(train_data)))
    # insert data into memory
    for file_path in os.listdir(train_data):
        buffer.append(generate_fictitious_episode(os.path.join(train_data, file_path))[0])

    optimizer = torch.optim.Adam(rssm.parameters(), lr=3e-4, eps=1e-4)

    for e in range(epochs):
        epoch_metrics = {
            "image_reconstruction_loss": 0.0,
            "reward_prediction_loss": 0.0,
            "kl_divergence_loss": 0.0,
            "total_loss": 0.0
        }
        for i in range(iterations_per_epoch):
            batch = buffer.sample(batch_size, tracelen=horizon, time_first=True)
            x, u, r, t = [torch.tensor(x).float().to(device) for x in batch]
            preprocess_img(x, depth=5)
            # encode image
            e_t = bottle(rssm.encoder, x)
            # compute the initial hidden and deterministic state
            h_t, s_t = rssm.get_init_state(e_t[0])
            # Initialize metrics
            kl_loss, rc_loss, re_loss = 0, 0, 0
            states, priors, posteriors, posterior_samples = [], [], [], []
            for i, a_t in enumerate(torch.unbind(u, dim=0)):
                a_t = torch.nn.functional.one_hot(a_t.long(), num_classes=action_size).squeeze().float().to(device)
                # predict next step of deterministic state
                h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
                states.append(h_t)
                # Compute predicted priors
                priors.append(rssm.state_prior(h_t))
                # Compute predicted posteriors
                posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
                # Sample from Gaussian
                posterior_samples.append(torch.distributions.Normal(*posteriors[-1]).rsample())
                # Advance state?
                s_t = posterior_samples[-1]
                # Compute states and posterior samples?
            states, posterior_samples = map(torch.stack, (states, posterior_samples))
            # Compute reconstruction loss
            decoded = bottle(rssm.decoder, states, posterior_samples)
            rec_loss = torch.nn.functional.mse_loss(decoded, x[1:],reduction='none').sum((2, 3, 4)).mean()
            # Compute KL divergence loss
            """kld_loss = torch.max(
                torch.distributions.kl.kl_divergence(posterior_dist, prior_dist).sum(-1),
                free_nats
            ).mean()"""
            prior_dist = torch.distributions.Normal(*map(torch.stack, zip(*priors)))
            posterior_dist = torch.distributions.Normal(*map(torch.stack, zip(*posteriors)))
            kld_loss = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist).sum(-1).mean()
            # Compute reward prediction loss
            predicted_reward = bottle(rssm.pred_reward, states, posterior_samples)
            rew_loss = torch.nn.functional.mse_loss(predicted_reward, r)

            loss = beta * kld_loss + rec_loss + rew_loss
            # Backprop
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(rssm.parameters(), 1000., norm_type=2)
            loss.backward()
            optimizer.step()

            epoch_metrics["image_reconstruction_loss"] += rec_loss.item()
            epoch_metrics["reward_prediction_loss"] += rew_loss.item()
            epoch_metrics["kl_divergence_loss"] += kld_loss.item()

        mean_rec_loss = epoch_metrics["image_reconstruction_loss"] / iterations_per_epoch
        mean_rew_loss = epoch_metrics["reward_prediction_loss"] / iterations_per_epoch
        mean_kld_loss = epoch_metrics["kl_divergence_loss"] / iterations_per_epoch
        print(f"[Epoch {e + 1}/{epochs}] | Reconstruction Loss: {mean_rec_loss:.5f} | Reward Loss: {mean_rew_loss:.5f} | KL Loss: {mean_kld_loss:.5f}")

        if (e + 1) % test_frequency == 0:
            test_batch = buffer.sample(1, tracelen=horizon, time_first=True)
            x, u, r, t = [torch.tensor(x).float().to(device) for x in test_batch]
            obs = x[0].unsqueeze(1)
            obs = preprocess_img(obs.float(), depth=5, inplace=False)
            e_t = bottle(rssm.encoder, obs)
            # compute the initial hidden and deterministic state
            h_t, s_t = rssm.get_init_state(e_t[0])
            decoded = []
            for i, a_t in enumerate(torch.unbind(u, dim=0)):
                with torch.no_grad():
                    a_t = torch.nn.functional.one_hot(a_t.long(), num_classes=action_size).squeeze(1).float().to(device)
                    # predict next step of deterministic state
                    h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
                    # Compute predicted priors
                    state_prior = rssm.state_prior(h_t)
                    new_e_t = torch.distributions.Normal(*state_prior).rsample()
                    # Compute predicted posteriors
                    state_posterior = rssm.state_posterior(h_t, new_e_t)
                    # Sample from Gaussian
                    s_t = torch.distributions.Normal(*state_posterior).rsample()
                    dec = rssm.decoder(h_t, s_t).squeeze().cpu().clamp_(-0.5, 0.5)
                    decoded.append(dec + 0.5)

            reshaped_x = x.squeeze(1).cpu().permute(0, 2, 3, 1) / 255.
            reshaped_decoded = torch.cat([torch.zeros_like(decoded[0]).unsqueeze(0), torch.cat([f.unsqueeze(0) for f in decoded], dim=0)]).permute(0, 2, 3, 1)
            img = np.hstack([reshaped_x.numpy(), reshaped_decoded.numpy()])
            fig, ax = plt.subplots(1, img.shape[0], figsize=(0.66 * (horizon + 1), 1.5))
            fig.suptitle(f"Epoch {e + 1}")
            for i in range(img.shape[0]):
                ax[i].imshow(img[i])
                ax[i].axis('off')
            plt.show()

    os.makedirs("models", exist_ok=True)
    torch.save(encoder.state_dict(), f"models/encoder_{epochs}.pth")
    torch.save(decoder.state_dict(), f"models/decoder_{epochs}.pth")
    torch.save(rssm.state_dict(), f"models/rssm_{epochs}.pth")
