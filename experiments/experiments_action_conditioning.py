import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from skimage.metrics import structural_similarity as ssim

from replay_buffer import ReplayBuffer
from vae import VAE
from rollout_buffer import RolloutBuffer
from utils import freeze_parameters


if __name__ == '__main__':
    # TODO: how to present these? Multibar plot? https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    num_samples = 20
    num_collected = 10
    num_actions = 9
    input_shape = (1, 3, 64, 64)
    latent_size = 128
    hidden_state_size = 256
    max_episode_len = 1500
    gamma = 0.99
    context_length = 1
    ssm_context_length = 4
    horizon = 20
    data_path = "super_tux_kart/trajectories"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    condition_on_action = True
    plot_samples = True
    overall_results_kl = {
        "rollout_one": {},
        "rollout_long": {},
        "replay_l2_one": {},
        "replay_l2_long": {},
        "replay_kl_one": {},
        "replay_kl_long": {},
        "rollout_one_no_act": {},
        "rollout_long_no_act": {},
        "replay_l2_one_no_act": {},
        "replay_l2_long_no_act": {},
        "replay_kl_one_no_act": {},
        "replay_kl_long_no_act": {}
    }
    overall_results_l1 = {
        "rollout_one": {},
        "rollout_long": {},
        "replay_l2_one": {},
        "replay_l2_long": {},
        "replay_kl_one": {},
        "replay_kl_long": {},
        "rollout_one_no_act": {},
        "rollout_long_no_act": {},
        "replay_l2_one_no_act": {},
        "replay_l2_long_no_act": {},
        "replay_kl_one_no_act": {},
        "replay_kl_long_no_act": {}
    }
    overall_results_ssim = {
        "rollout_one": {},
        "rollout_long": {},
        "replay_l2_one": {},
        "replay_l2_long": {},
        "replay_kl_one": {},
        "replay_kl_long": {},
        "rollout_one_no_act": {},
        "rollout_long_no_act": {},
        "replay_l2_one_no_act": {},
        "replay_l2_long_no_act": {},
        "replay_kl_one_no_act": {},
        "replay_kl_long_no_act": {}
    }

    for k in overall_results_kl.keys():
        overall_results_kl[k]["mean"] = np.zeros(shape=(horizon,))
        overall_results_kl[k]["std"] = np.zeros(shape=(horizon,))
        overall_results_l1[k]["mean"] = np.zeros(shape=(horizon,))
        overall_results_l1[k]["std"] = np.zeros(shape=(horizon,))
        overall_results_ssim[k]["mean"] = np.zeros(shape=(horizon,))
        overall_results_ssim[k]["std"] = np.zeros(shape=(horizon,))

    track_list = ["fortmagma", "lighthouse", "snes_rainbowroad", "snowmountain", "volcano_island"]
    for track_name in track_list:
        traj_path = f"{data_path}/{track_name}/train"
        test_traj_path = f"{data_path}/{track_name}/test"

        rollout_world_model = RolloutBuffer(
            num_collected=num_collected,
            obs_size=latent_size,
            num_actions=num_actions,
            max_episode_len=max_episode_len,
            gamma=gamma,
            context_length=context_length
        )

        replay_world_model = ReplayBuffer(
            capacity=num_collected * max_episode_len,
            obs_size=latent_size,
            num_actions=num_actions
        )

        vae = VAE(input_shape=input_shape, latent_size=latent_size)
        vae.load_state_dict(torch.load(f"vae_models/super_tux_kart/{track_name}/encoder_{latent_size}_250_image.pth"))
        vae.to("cuda")
        vae.eval()
        freeze_parameters(vae)

        for traj in os.listdir(traj_path)[:num_collected]:
            data = np.load(f"{traj_path}/{traj}", allow_pickle=True)
            latents, means, logvars = vae.get_latent(torch.tensor(data["observations"] / 255.).float().permute(0, 3, 1, 2).to(device))
            actions = torch.from_numpy(data["actions"]).float().unsqueeze(-1)
            latents_next = torch.cat([latents[1:], torch.zeros(size=(1, latent_size)).to(device)], dim=0)
            rollout_world_model.add_trajectory(
                latents,
                actions,
                latents_next,
                means,
                logvars
            )
            for l, a, l_next, m, lg in zip(latents, actions, latents_next, means, logvars):
                replay_world_model.add_transition(l, a, l_next, m, lg)


        test_traj = np.load(f"{test_traj_path}/{os.listdir(test_traj_path)[-1]}", allow_pickle=True)
        test_traj = {
            "observations": test_traj["observations"],
            "actions": test_traj["actions"],
            "next_observations": np.concatenate([test_traj["observations"][1:], np.zeros(shape=(1, 64, 64, 3))], axis=0),
        }

        kl_divergence_with_base = {
            "rollout_one": [],
            "rollout_long": [],
            "replay_l2_one": [],
            "replay_l2_long": [],
            "replay_kl_one": [],
            "replay_kl_long": [],
            "rollout_one_no_act": [],
            "rollout_long_no_act": [],
            "replay_l2_one_no_act": [],
            "replay_l2_long_no_act": [],
            "replay_kl_one_no_act": [],
            "replay_kl_long_no_act": []
        }
        l1_distances_with_base = {
            "rollout_one": [],
            "rollout_long": [],
            "replay_l2_one": [],
            "replay_l2_long": [],
            "replay_kl_one": [],
            "replay_kl_long": [],
            "rollout_one_no_act": [],
            "rollout_long_no_act": [],
            "replay_l2_one_no_act": [],
            "replay_l2_long_no_act": [],
            "replay_kl_one_no_act": [],
            "replay_kl_long_no_act": []
        }
        ssim_with_base = {
            "rollout_one": [],
            "rollout_long": [],
            "replay_l2_one": [],
            "replay_l2_long": [],
            "replay_kl_one": [],
            "replay_kl_long": [],
            "rollout_one_no_act": [],
            "rollout_long_no_act": [],
            "replay_l2_one_no_act": [],
            "replay_l2_long_no_act": [],
            "replay_kl_one_no_act": [],
            "replay_kl_long_no_act": []
        }
        for s in range(num_samples):
            print(f"[{track_name}] Sample {s+1}/{num_samples}")
            tr_idx = np.random.randint(ssm_context_length, test_traj["observations"].shape[0] - horizon - 1)

            # Real
            real_seq = []
            real_mu_seq = []
            real_std_seq = []
            # Baseline long-term
            ssm_long_term_seq = []
            # Rollout search
            rollout_one_seq = []
            rollout_long_seq = []
            # Replay search, L2
            replay_l2_one_seq = []
            replay_l2_long_seq = []
            # Replay search, KL
            replay_kl_one_seq = []
            replay_kl_long_seq = []
            replay_kl_div_one = []
            replay_kl_div_long = []
            # Rollout search - no conditioning
            rollout_one_seq_no_act = []
            rollout_long_seq_no_act = []
            # Replay search, L2 - no conditioning
            replay_l2_one_seq_no_act = []
            replay_l2_long_seq_no_act = []
            # Replay search, KL - no conditioning
            replay_kl_one_seq_no_act = []
            replay_kl_long_seq_no_act = []
            replay_kl_div_one_no_act = []
            replay_kl_div_long_no_act = []

            # DISCLAIMER: First step is done outside the for loop to retain indices for long-term search.

            # Real sequence
            real_seq.append(test_traj["observations"][tr_idx] / 255.)
            latent, mu, logvar = vae.get_latent(torch.tensor(test_traj["observations"][tr_idx] / 255.).float().unsqueeze(0).permute(0, 3, 1, 2).to(device))
            real_mu_seq.append(mu)
            real_std_seq.append(torch.sqrt(torch.exp(logvar)))
            # Rollout 1-step
            similar_transition, similar_indices_rollout = rollout_world_model.sample_similar(latent.cpu(), return_indices=True, action=test_traj["actions"][tr_idx])
            similar_transition_no_act, similar_indices_rollout_no_act = rollout_world_model.sample_similar(latent.cpu(), return_indices=True, action=None)
            rollout_one_seq.append(similar_transition.observation)
            rollout_one_seq_no_act.append(similar_transition_no_act.observation)
            # Rollout long-term
            imagined_obs = []
            for ts, tr in enumerate(similar_indices_rollout.squeeze().numpy()):
                imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr]))
                if int(imagined_next.action) == int(test_traj["actions"][tr_idx]):
                    imagined_obs.append(imagined_next.observation)
            # To avoid extreme estimations, we approximate without action conditioning whenever there are too few samples with the same action
            if len(imagined_obs) <= num_collected / 2:
                imagined_obs = []
                for ts, tr in enumerate(similar_indices_rollout.squeeze().numpy()):
                    imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr]))
                    imagined_obs.append(imagined_next.observation)
            rollout_long_seq.append(torch.cat(imagined_obs))
            # No act
            imagined_obs_no_act = []
            for ts, tr in enumerate(similar_indices_rollout_no_act.squeeze().numpy()):
                imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr]))
                if int(imagined_next.action) == int(test_traj["actions"][tr_idx]):
                    imagined_obs_no_act.append(imagined_next.observation)
            # To avoid extreme estimations, we approximate without action conditioning whenever there are too few samples with the same action
            if len(imagined_obs_no_act) <= num_collected / 2:
                imagined_obs_no_act = []
                for ts, tr in enumerate(similar_indices_rollout.squeeze().numpy()):
                    imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr]))
                    imagined_obs_no_act.append(imagined_next.observation)
            rollout_long_seq_no_act.append(torch.cat(imagined_obs_no_act))
            # Replay L2 1-step
            similar_transition_replay, similar_indices_replay = replay_world_model.sample_similar(latent.cpu(), action=test_traj["actions"][tr_idx], k=10)
            similar_transition_replay_no_act, similar_indices_replay_no_act = replay_world_model.sample_similar(latent.cpu(), action=None, k=10)
            obs = similar_transition_replay.observation
            replay_l2_one_seq.append(obs)
            # Replay L2 long-term
            replay_l2_long_seq.append(obs)
            # No act
            obs = similar_transition_replay_no_act.observation
            replay_l2_one_seq_no_act.append(obs)
            replay_l2_long_seq_no_act.append(obs)
            # Replay KL 1-step
            similar_transition, similar_indices_kl, kl_div = replay_world_model.sample_similar_distribution(mu.cpu(), logvar.cpu(), action=test_traj["actions"][tr_idx])
            similar_transition_no_act, similar_indices_kl_no_act, kl_div_no_act = replay_world_model.sample_similar_distribution(mu.cpu(), logvar.cpu(), action=None)
            obs = torch.distributions.Normal(similar_transition.mean, similar_transition.sigma).rsample()
            replay_kl_one_seq.append(obs)
            replay_kl_div_one.append(kl_div.item())
            # Replay KL long-term
            replay_kl_long_seq.append(obs)
            replay_kl_div_long.append(kl_div.item())
            # No act
            obs = torch.distributions.Normal(similar_transition_no_act.mean, similar_transition_no_act.sigma).rsample()
            replay_kl_one_seq_no_act.append(obs)
            replay_kl_long_seq_no_act.append(obs)
            replay_kl_div_one_no_act.append(kl_div_no_act.item())
            replay_kl_div_long_no_act.append(kl_div_no_act.item())

            for i in range(1, horizon):
                # Base
                real_seq.append(test_traj["observations"][tr_idx + i] / 255.)
                latent, mu, logvar = vae.get_latent(torch.tensor(test_traj["observations"][tr_idx + i] / 255.).float().unsqueeze(0).permute(0, 3, 1, 2).to(device))
                real_mu_seq.append(mu)
                real_std_seq.append(torch.sqrt(torch.exp(logvar)))
                # One-step search-based
                similar_transition, _ = rollout_world_model.sample_similar(latent.cpu(), return_indices=False, action=test_traj["actions"][tr_idx + i])
                similar_transition_no_act, _ = rollout_world_model.sample_similar(latent.cpu(), return_indices=False, action=None)
                rollout_one_seq.append(similar_transition.next_observation)
                rollout_one_seq_no_act.append(similar_transition_no_act.next_observation)
                # Long-term search-based
                imagined_obs = []
                # Search best sample for each trajectory
                for ts, tr in enumerate(similar_indices_rollout.squeeze().numpy()):
                    imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr + i]))
                    if int(imagined_next.action) == int(test_traj["actions"][tr_idx + i]):
                        imagined_obs.append(imagined_next.observation)
                # To avoid extreme estimations, we approximate without action conditioning whenever there are too few samples with the same action
                if len(imagined_obs) <= num_collected / 2:
                    imagined_obs = []
                    for ts, tr in enumerate(similar_indices_rollout.squeeze().numpy()):
                        imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr + i]))
                        imagined_obs.append(imagined_next.observation)
                rollout_long_seq.append(torch.cat(imagined_obs))
                # No act
                imagined_obs_no_act = []
                # Search best sample for each trajectory
                for ts, tr in enumerate(similar_indices_rollout_no_act.squeeze().numpy()):
                    imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr + i]))
                    if int(imagined_next.action) == int(test_traj["actions"][tr_idx + i]):
                        imagined_obs_no_act.append(imagined_next.observation)
                # To avoid extreme estimations, we approximate without action conditioning whenever there are too few samples with the same action
                if len(imagined_obs) <= num_collected / 2:
                    imagined_obs_no_act = []
                    for ts, tr in enumerate(similar_indices_rollout.squeeze().numpy()):
                        imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr + i]))
                        imagined_obs_no_act.append(imagined_next.observation)
                rollout_long_seq_no_act.append(torch.cat(imagined_obs))
                # Replay L2 1-step
                similar_transition, _ = replay_world_model.sample_similar(latent.cpu(), action=test_traj["actions"][tr_idx + i], k=10)
                similar_transition_no_act, _ = replay_world_model.sample_similar(latent.cpu(), action=None, k=10)
                obs = similar_transition.next_observation
                replay_l2_one_seq.append(obs)
                # No act
                obs = similar_transition_no_act.next_observation
                replay_l2_one_seq_no_act.append(obs)
                # Replay L2 long-term
                imagined_obs = []
                for tr in similar_indices_replay.squeeze().numpy():
                    imagined_next = replay_world_model.get_transition(torch.tensor([tr + i]))
                    imagined_obs.append(imagined_next.observation)
                replay_l2_long_seq.append(torch.cat(imagined_obs))
                # No act
                for tr in similar_indices_replay_no_act.squeeze().numpy():
                    imagined_next = replay_world_model.get_transition(torch.tensor([tr + i]))
                    imagined_obs.append(imagined_next.observation)
                replay_l2_long_seq_no_act.append(torch.cat(imagined_obs))
                # Replay KL 1-step
                _, next_mu, next_logvar = vae.get_latent(torch.tensor(test_traj["next_observations"][tr_idx + i] / 255.).float().unsqueeze(0).permute(0, 3, 1, 2).to(device))
                similar_transition, _, kl_div = replay_world_model.sample_similar_distribution(next_mu.cpu(), next_logvar.cpu(), action=test_traj["actions"][tr_idx + i])
                similar_transition_no_act, _, kl_div_no_act = replay_world_model.sample_similar_distribution(next_mu.cpu(), next_logvar.cpu(), action=None)
                obs = torch.distributions.Normal(similar_transition.mean, similar_transition.sigma).rsample()
                replay_kl_one_seq.append(obs)
                replay_kl_div_one.append(kl_div.item())
                # No act
                obs = torch.distributions.Normal(similar_transition_no_act.mean, similar_transition.sigma).rsample()
                replay_kl_one_seq_no_act.append(obs)
                replay_kl_div_one_no_act.append(kl_div.item())
                # Replay KL long-term
                imagined_next = replay_world_model.get_transition(torch.tensor([similar_indices_kl + i]))
                ref_distrib = torch.distributions.Normal(mu.cpu(), torch.sqrt(torch.exp(logvar)).cpu())
                next_distrib = torch.distributions.Normal(imagined_next.mean, imagined_next.sigma)
                obs_next = next_distrib.rsample()
                replay_kl_long_seq.append(obs_next.squeeze())
                replay_kl_div_long.append(torch.distributions.kl.kl_divergence(ref_distrib, next_distrib).sum(-1).item())
                # No act
                imagined_next = replay_world_model.get_transition(torch.tensor([similar_indices_kl_no_act + i]))
                next_distrib = torch.distributions.Normal(imagined_next.mean, imagined_next.sigma)
                obs_next = next_distrib.rsample()
                replay_kl_long_seq_no_act.append(obs_next.squeeze())
                replay_kl_div_long_no_act.append(torch.distributions.kl.kl_divergence(ref_distrib, next_distrib).sum(-1).item())

            # Base distribution
            dist_base = torch.distributions.normal.Normal(torch.cat(real_mu_seq), torch.cat(real_std_seq))
            base_seq_tensor = torch.cat([torch.tensor(x).unsqueeze(0).permute(0, 3, 1, 2) for x in real_seq]).to(device)
            # One-step search-based
            mean_rollout_one = torch.cat([x.mean(1) for x in rollout_one_seq], dim=0)
            std_rollout_one = torch.cat([x.std(1) for x in rollout_one_seq], dim=0)
            dist_rollout_one = torch.distributions.normal.Normal(mean_rollout_one.to(device), std_rollout_one.to(device))
            sample_rollout_one = dist_rollout_one.sample()
            decoded_rollout_one = vae.decode(sample_rollout_one.to(device))
            kl_divergence_with_base["rollout_one"].append(torch.distributions.kl.kl_divergence(dist_base, dist_rollout_one).sum(-1).unsqueeze(0))
            l1_distances_with_base["rollout_one"].append(torch.nn.functional.l1_loss(decoded_rollout_one, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["rollout_one"].append(torch.tensor([ssim(decoded_rollout_one[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in range(horizon)]).unsqueeze(0))
            # Long-term search-based
            mean_rollout_long = torch.cat([x.mean(0).unsqueeze(0) for x in rollout_long_seq], dim=0)
            std_rollout_long = torch.cat([x.std(0).unsqueeze(0) for x in rollout_long_seq], dim=0)
            dist_rollout_long = torch.distributions.normal.Normal(mean_rollout_long.to(device), std_rollout_long.to(device))
            sample_rollout_long = dist_rollout_long.sample()
            decoded_rollout_long = vae.decode(sample_rollout_long.to(device))
            kl_divergence_with_base["rollout_long"].append(torch.distributions.kl.kl_divergence(dist_base, dist_rollout_long).sum(-1).unsqueeze(0))
            l1_distances_with_base["rollout_long"].append(torch.nn.functional.l1_loss(decoded_rollout_long, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["rollout_long"].append(
                torch.tensor([ssim(decoded_rollout_long[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in range(horizon)]).unsqueeze(0))
            # Replay L2 1-step
            mean_replay_l2_one = torch.stack([x.mean(0) for x in replay_l2_one_seq], dim=0)
            std_replay_l2_one = torch.stack([x.std(0) for x in replay_l2_one_seq], dim=0)
            dist_replay_l2_one = torch.distributions.normal.Normal(mean_replay_l2_one.to(device), std_replay_l2_one.to(device))
            sample_obs = dist_replay_l2_one.sample()
            decoded_replay_l2_one = vae.decode(sample_obs.to(device))
            kl_divergence_with_base["replay_l2_one"].append(torch.distributions.kl.kl_divergence(dist_base, dist_replay_l2_one).sum(-1).unsqueeze(0))
            l1_distances_with_base["replay_l2_one"].append(torch.nn.functional.l1_loss(decoded_replay_l2_one, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["replay_l2_one"].append(
                torch.tensor([ssim(decoded_replay_l2_one[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in range(horizon)]).unsqueeze(0))
            # Replay L2 long-term
            mean_replay_l2_long = torch.stack([x.mean(0) for x in replay_l2_long_seq], dim=0)
            std_replay_l2_long = torch.stack([x.std(0) for x in replay_l2_long_seq], dim=0)
            dist_replay_l2_long = torch.distributions.normal.Normal(mean_replay_l2_long.to(device), std_replay_l2_long.to(device))
            sample_replay_l2_long = dist_replay_l2_long.sample()
            decoded_replay_l2_long = vae.decode(sample_replay_l2_long.to(device))
            kl_divergence_with_base["replay_l2_long"].append(torch.distributions.kl.kl_divergence(dist_base, dist_replay_l2_long).sum(-1).unsqueeze(0))
            l1_distances_with_base["replay_l2_long"].append(torch.nn.functional.l1_loss(decoded_replay_l2_long, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["replay_l2_long"].append(
                torch.tensor(
                    [ssim(decoded_replay_l2_long[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in range(horizon)]).unsqueeze(0))
            # Replay KL 1-step
            decoded_replay_kl_one = vae.decode(torch.stack(replay_kl_one_seq).to(device))
            kl_divergence_with_base["replay_kl_one"].append(torch.tensor(replay_kl_div_one).unsqueeze(0))
            l1_distances_with_base["replay_kl_one"].append(torch.nn.functional.l1_loss(decoded_replay_kl_one, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["replay_kl_one"].append(
                torch.tensor([ssim(decoded_replay_kl_one[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in range(horizon)]).unsqueeze(0))
            # Replay KL long-term
            decoded_replay_kl_long = vae.decode(torch.stack(replay_kl_long_seq).to(device))
            kl_divergence_with_base["replay_kl_long"].append(torch.tensor(replay_kl_div_long).unsqueeze(0))
            l1_distances_with_base["replay_kl_long"].append(torch.nn.functional.l1_loss(decoded_replay_kl_long, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["replay_kl_long"].append(
                torch.tensor(
                    [ssim(decoded_replay_kl_long[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in range(horizon)]).unsqueeze(0))
            # NO ACT One-step search-based
            mean_rollout_one_no_act = torch.cat([x.mean(1) for x in rollout_one_seq_no_act], dim=0)
            std_rollout_one_no_act = torch.cat([x.std(1) for x in rollout_one_seq_no_act], dim=0)
            dist_rollout_one_no_act = torch.distributions.normal.Normal(mean_rollout_one_no_act.to(device), std_rollout_one_no_act.to(device))
            sample_rollout_one_no_act = dist_rollout_one_no_act.sample()
            decoded_rollout_one_no_act = vae.decode(sample_rollout_one_no_act.to(device))
            kl_divergence_with_base["rollout_one_no_act"].append(torch.distributions.kl.kl_divergence(dist_base, dist_rollout_one_no_act).sum(-1).unsqueeze(0))
            l1_distances_with_base["rollout_one_no_act"].append(torch.nn.functional.l1_loss(decoded_rollout_one_no_act, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["rollout_one_no_act"].append(torch.tensor(
                [ssim(decoded_rollout_one_no_act[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in range(horizon)]).unsqueeze(0))
            # NO ACT Long-term search-based
            mean_rollout_long_no_act = torch.cat([x.mean(0).unsqueeze(0) for x in rollout_long_seq_no_act], dim=0)
            std_rollout_long_no_act = torch.cat([x.std(0).unsqueeze(0) for x in rollout_long_seq_no_act], dim=0)
            dist_rollout_long_no_act = torch.distributions.normal.Normal(mean_rollout_long_no_act.to(device), std_rollout_long_no_act.to(device))
            sample_rollout_long_no_act = dist_rollout_long_no_act.sample()
            decoded_rollout_long_no_act = vae.decode(sample_rollout_long_no_act.to(device))
            kl_divergence_with_base["rollout_long_no_act"].append(torch.distributions.kl.kl_divergence(dist_base, dist_rollout_long_no_act).sum(-1).unsqueeze(0))
            l1_distances_with_base["rollout_long_no_act"].append(torch.nn.functional.l1_loss(decoded_rollout_long_no_act, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["rollout_long_no_act"].append(
                torch.tensor([ssim(decoded_rollout_long_no_act[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in
                              range(horizon)]).unsqueeze(0))
            # NO ACT Replay L2 1-step
            mean_replay_l2_one_no_act = torch.stack([x.mean(0) for x in replay_l2_one_seq_no_act], dim=0)
            std_replay_l2_one_no_act = torch.stack([x.std(0) for x in replay_l2_one_seq_no_act], dim=0)
            dist_replay_l2_one_no_act = torch.distributions.normal.Normal(mean_replay_l2_one_no_act.to(device), std_replay_l2_one_no_act.to(device))
            sample_obs_no_act = dist_replay_l2_one_no_act.sample()
            decoded_replay_l2_one_no_act = vae.decode(sample_obs_no_act.to(device))
            kl_divergence_with_base["replay_l2_one_no_act"].append(torch.distributions.kl.kl_divergence(dist_base, dist_replay_l2_one_no_act).sum(-1).unsqueeze(0))
            l1_distances_with_base["replay_l2_one_no_act"].append(torch.nn.functional.l1_loss(decoded_replay_l2_one_no_act, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["replay_l2_one_no_act"].append(
                torch.tensor([ssim(decoded_replay_l2_one_no_act[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in
                              range(horizon)]).unsqueeze(0))
            # NO ACT Replay L2 long-term
            mean_replay_l2_long_no_act = torch.stack([x.mean(0) for x in replay_l2_long_seq_no_act], dim=0)
            std_replay_l2_long_no_act = torch.stack([x.std(0) for x in replay_l2_long_seq_no_act], dim=0)
            dist_replay_l2_long_no_act = torch.distributions.normal.Normal(mean_replay_l2_long_no_act.to(device), std_replay_l2_long_no_act.to(device))
            sample_replay_l2_long_no_act = dist_replay_l2_long_no_act.sample()
            decoded_replay_l2_long_no_act = vae.decode(sample_replay_l2_long_no_act.to(device))
            kl_divergence_with_base["replay_l2_long_no_act"].append(torch.distributions.kl.kl_divergence(dist_base, dist_replay_l2_long_no_act).sum(-1).unsqueeze(0))
            l1_distances_with_base["replay_l2_long_no_act"].append(torch.nn.functional.l1_loss(decoded_replay_l2_long_no_act, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["replay_l2_long_no_act"].append(
                torch.tensor(
                    [ssim(decoded_replay_l2_long_no_act[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in
                     range(horizon)]).unsqueeze(0))
            # NO ACT Replay KL 1-step
            decoded_replay_kl_one_no_act = vae.decode(torch.stack(replay_kl_one_seq_no_act).to(device))
            kl_divergence_with_base["replay_kl_one_no_act"].append(torch.tensor(replay_kl_div_one_no_act).unsqueeze(0))
            l1_distances_with_base["replay_kl_one_no_act"].append(torch.nn.functional.l1_loss(decoded_replay_kl_one_no_act, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["replay_kl_one_no_act"].append(
                torch.tensor([ssim(decoded_replay_kl_one_no_act[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in
                              range(horizon)]).unsqueeze(0))
            # NO ACT Replay KL long-term
            decoded_replay_kl_long_no_act = vae.decode(torch.stack(replay_kl_long_seq_no_act).to(device))
            kl_divergence_with_base["replay_kl_long_no_act"].append(torch.tensor(replay_kl_div_long_no_act).unsqueeze(0))
            l1_distances_with_base["replay_kl_long_no_act"].append(torch.nn.functional.l1_loss(decoded_replay_kl_long_no_act, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["replay_kl_long_no_act"].append(
                torch.tensor(
                    [ssim(decoded_replay_kl_long_no_act[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in
                     range(horizon)]).unsqueeze(0))
            # Plot samples
            # 1-step
            if plot_samples:
                os.makedirs(f"experiments/action_conditioning/super_tux_kart/{track_name}/samples/one_step/", exist_ok=True)
                np.savez_compressed(
                    f"experiments/action_conditioning/super_tux_kart/{track_name}/samples/one_step/decoded_one_step_sample_{s}.npz",
                    real=np.array([cv2.resize(x, (128, 128)) for x in real_seq]),
                    rollout=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_rollout_one]),
                    rollout_no_act=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_rollout_one_no_act]),
                    replay_l2=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_l2_one]),
                    replay_l2_no_act=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_l2_one_no_act]),
                    replay_kl=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_kl_one]),
                    replay_kl_no_act=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_kl_one_no_act])
                )
                os.makedirs(f"experiments/action_conditioning/super_tux_kart/{track_name}/samples/long_term/", exist_ok=True)
                np.savez_compressed(
                    f"experiments/action_conditioning/super_tux_kart/{track_name}/samples/long_term/decoded_long_term_sample_{s}.npz",
                    real=np.array([cv2.resize(x, (128, 128)) for x in real_seq]),
                    rollout=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_rollout_long]),
                    rollout_no_act=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_rollout_long_no_act]),
                    replay_l2=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_l2_long]),
                    replay_l2_no_act=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_l2_long_no_act]),
                    replay_kl=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_kl_long]),
                    replay_kl_no_act=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_kl_long_no_act])
                )

        # Stat plots
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(17, 7), sharex=True)
        handles = []
        colors = ["tab:red", "salmon", "tab:blue", "lightblue", "tab:green", "lightgreen"]
        offsets = [0, 2, 4]
        width = 0.15
        multiplier = 1
        x_offsets = np.arange(horizon)
        for off, k in zip(offsets, ["rollout", "replay_l2", "replay_kl"]):
            color_offset = 0
            for suffix in ["", "_no_act"]:
                # Rollout one-step
                mean_kl_dist = torch.cat(kl_divergence_with_base[f"{k}_one{suffix}"], dim=0).mean(dim=0).cpu().numpy()
                std_kl_dist = torch.cat(kl_divergence_with_base[f"{k}_one{suffix}"], dim=0).std(dim=0).cpu().numpy()
                mean_l1_dist = torch.cat(l1_distances_with_base[f"{k}_one{suffix}"], dim=0).mean(dim=0).cpu().numpy()
                std_l1_dist = torch.cat(l1_distances_with_base[f"{k}_one{suffix}"], dim=0).std(dim=0).cpu().numpy()
                mean_ssim_dist = torch.cat(ssim_with_base[f"{k}_one{suffix}"], dim=0).mean(dim=0).cpu().numpy()
                std_ssim_dist = torch.cat(ssim_with_base[f"{k}_one{suffix}"], dim=0).std(dim=0).cpu().numpy()
                # Save overall results
                overall_results_kl[f"{k}_one{suffix}"]["mean"] += mean_kl_dist
                overall_results_kl[f"{k}_one{suffix}"]["std"] += std_kl_dist
                overall_results_l1[f"{k}_one{suffix}"]["mean"] += mean_l1_dist
                overall_results_l1[f"{k}_one{suffix}"]["std"] += std_l1_dist
                overall_results_ssim[f"{k}_one{suffix}"]["mean"] += mean_ssim_dist
                overall_results_ssim[f"{k}_one{suffix}"]["std"] += std_ssim_dist
                # KL
                handles.append(ax[0].bar(x_offsets + width * multiplier, mean_kl_dist, width, label="Search-based (1-step)", color=colors[off + color_offset]))
                lowers = np.max([np.zeros(shape=(mean_kl_dist.shape[0]),), mean_kl_dist - std_kl_dist], axis=0)
                ax[0].errorbar(x_offsets + width * multiplier, mean_kl_dist, yerr=[lowers, std_kl_dist], capsize=2, ls='none', ecolor='black', elinewidth=0.5)
                # Search long-term
                mean_kl_dist = torch.cat(kl_divergence_with_base[f"{k}_long{suffix}"], dim=0).mean(dim=0).cpu().numpy()
                std_kl_dist = torch.cat(kl_divergence_with_base[f"{k}_long{suffix}"], dim=0).std(dim=0).cpu().numpy()
                # Save overall results
                overall_results_kl[f"{k}_long{suffix}"]["mean"] += mean_kl_dist
                overall_results_kl[f"{k}_long{suffix}"]["std"] += std_kl_dist
                # KL
                lowers = np.max([np.zeros(shape=(mean_kl_dist.shape[0]), ), mean_kl_dist - std_kl_dist], axis=0)
                ax[1].bar(x_offsets + width * multiplier, mean_kl_dist, width, label="Search-based (long-term)", color=colors[off + color_offset])
                ax[1].errorbar(x_offsets + width * multiplier, mean_kl_dist, yerr=[lowers, std_kl_dist], capsize=2, ls='none', ecolor='black', elinewidth=0.5)

                color_offset += 1
                multiplier += 1

            # General
            ax[0].set_ylabel("KL Divergence")
            ax[1].set_ylabel("KL Divergence")
            ax[1].set_xlabel("Timestep")

            ax[0].set_xticks(x_offsets + 3 * width)
            ax[0].set_yticks([0, 100, 200, 300, 400])
            ax[1].set_xticks(x_offsets + 3 * width)
            ax[1].set_yticks([0, 150, 300, 450, 600])

            ax[0].set_xticklabels(x_offsets)
            ax[0].set_yticklabels([0, 100, 200, 300, 400])
            ax[1].set_xticklabels(x_offsets)
            ax[1].set_yticklabels([0, 150, 300, 450, 600])

            ax[0].set_ylim(0, 400)
            ax[1].set_ylim(0, 600)

        fig.legend(handles, ["Rollout", "Rollout w/o action", "Replay-L2", "Replay L2 w/o action", "Replay-KL", "Replay-KL w/o action"], loc="upper right")
        plt.tight_layout()
        fig.savefig(f"experiments/action_conditioning/super_tux_kart/{track_name}/numerical_comparison.png")
        fig.clear()

    # Benchmark results
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(17, 7), sharex=True)
    for k in overall_results_kl.keys():
        overall_results_kl[k]["mean"] = overall_results_kl[k]["mean"] / float(len(track_list))
        overall_results_kl[k]["std"] = overall_results_kl[k]["std"] / float(len(track_list))
        overall_results_l1[k]["mean"] = overall_results_l1[k]["mean"] / float(len(track_list))
        overall_results_l1[k]["std"] = overall_results_l1[k]["std"] / float(len(track_list))
        overall_results_ssim[k]["mean"] = overall_results_ssim[k]["mean"] / float(len(track_list))
        overall_results_ssim[k]["std"] = overall_results_ssim[k]["std"] / float(len(track_list))

    handles = []
    multiplier = 1
    for off, k in zip(offsets, ["rollout", "replay_l2", "replay_kl"]):
        color_offset = 0
        for suffix in ["", "_no_act"]:
            # Rollout 1-step
            mean_kl_dist = overall_results_kl[f"{k}_one{suffix}"]["mean"]
            std_kl_dist = overall_results_kl[f"{k}_one{suffix}"]["std"]
            lowers = np.max([np.zeros(shape=(mean_kl_dist.shape[0]), ), mean_kl_dist - std_kl_dist], axis=0)
            handles.append(ax[0].bar(x_offsets + width * multiplier, mean_kl_dist, width, label="Search-based (1-step)", color=colors[off + color_offset]))
            ax[0].errorbar(x_offsets + width * multiplier, mean_kl_dist, yerr=std_kl_dist, capsize=2, ls='none', ecolor='black', elinewidth=0.5)
            # Search long-term
            mean_kl_dist = torch.cat(kl_divergence_with_base[f"{k}_long{suffix}"], dim=0).mean(dim=0).cpu().numpy()
            std_kl_dist = torch.cat(kl_divergence_with_base[f"{k}_long{suffix}"], dim=0).std(dim=0).cpu().numpy()
            lowers = np.max([np.zeros(shape=(mean_kl_dist.shape[0]), ), mean_kl_dist - std_kl_dist], axis=0)
            # KL
            ax[1].bar(x_offsets + width * multiplier, mean_kl_dist, width, label="Search-based (long-term)", color=colors[off + color_offset])
            ax[1].errorbar(x_offsets + width * multiplier, mean_kl_dist, yerr=std_kl_dist, capsize=2, ls='none', ecolor='black', elinewidth=0.5)

            color_offset += 1
            multiplier += 1

    # General
    ax[0].set_ylabel("KL Divergence", fontsize=20)
    ax[1].set_ylabel("KL Divergence", fontsize=20)
    ax[1].set_xlabel("Timestep", fontsize=20)

    ax[0].set_xticks(x_offsets + 3 * width)
    ax[0].set_yticks([0, 100, 200, 300, 400])
    ax[1].set_xticks(x_offsets + 3 * width)
    ax[1].set_yticks([0, 150, 300, 450, 600])

    ax[0].set_xticklabels(x_offsets, fontsize=18)
    ax[0].set_yticklabels([0, 100, 200, 300, 400], fontsize=18)
    ax[1].set_xticklabels(x_offsets, fontsize=18)
    ax[1].set_yticklabels([0, 150, 300, 450, 600], fontsize=18)

    ax[0].set_ylim(0, 400)
    ax[1].set_ylim(0, 600)

    fig.legend(handles, ["Rollout", "Rollout - no act.", "Replay-L2", "Replay L2 - no act.", "Replay-KL", "Replay-KL - no act."], loc="upper right", fontsize=15)
    fig.savefig(f"experiments/action_conditioning/super_tux_kart/benchmark_numerical_comparison.png", bbox_inches='tight')
    fig.clear()
