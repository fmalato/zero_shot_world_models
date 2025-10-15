import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from matplotlib.lines import Line2D
from skimage.metrics import structural_similarity as ssim

from replay_buffer import ReplayBuffer
from vae import VAE
from ssm import NextStepPredictionSSMMineRL
from rollout_buffer import RolloutBuffer
from utils import freeze_parameters


if __name__ == '__main__':
    num_samples = 20
    num_collected = 10
    num_actions = 10
    input_shape = (1, 3, 64, 64)
    latent_size = 512
    hidden_state_size = 256
    max_episode_len = 5000
    gamma = 0.99
    context_length = 1
    ssm_context_length = 4
    horizon = 20
    data_path = "minerl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    condition_on_action = False
    overall_results_kl = {
        "rollout_one": {},
        "rollout_long": {},
        "replay_l2_one": {},
        "replay_l2_long": {},
        "replay_kl_one": {},
        "replay_kl_long": {},
        "ssm_one": {},
        "ssm_long": {}
    }
    overall_results_l1 = {
        "rollout_one": {},
        "rollout_long": {},
        "replay_l2_one": {},
        "replay_l2_long": {},
        "replay_kl_one": {},
        "replay_kl_long": {},
        "ssm_one": {},
        "ssm_long": {}
    }
    overall_results_ssim = {
        "rollout_one": {},
        "rollout_long": {},
        "replay_l2_one": {},
        "replay_l2_long": {},
        "replay_kl_one": {},
        "replay_kl_long": {},
        "ssm_one": {},
        "ssm_long": {}
    }

    for k in overall_results_kl.keys():
        overall_results_kl[k]["mean"] = np.zeros(shape=(horizon,))
        overall_results_kl[k]["std"] = np.zeros(shape=(horizon,))
        overall_results_l1[k]["mean"] = np.zeros(shape=(horizon,))
        overall_results_l1[k]["std"] = np.zeros(shape=(horizon,))
        overall_results_ssim[k]["mean"] = np.zeros(shape=(horizon,))
        overall_results_ssim[k]["std"] = np.zeros(shape=(horizon,))

    env_names = ["MineRLTreechop-v0", "MineRLNavigate-v0"]
    for track_name in env_names:
        traj_path = f"{data_path}/{track_name}"
        test_traj_path = f"{data_path}/{track_name}"

        os.makedirs(f"experiments/prediction_horizon/minerl/{track_name}/samples/one_step/", exist_ok=True)
        os.makedirs(f"experiments/prediction_horizon/minerl/{track_name}/samples/long_term/", exist_ok=True)

        rollout_world_model = RolloutBuffer(
            num_collected=num_collected,
            obs_size=latent_size,
            num_actions=num_actions,
            max_episode_len=max_episode_len,
            gamma=gamma,
            context_length=context_length,
            is_minerl=True
        )

        replay_world_model = ReplayBuffer(
            capacity=num_collected * max_episode_len,
            obs_size=latent_size,
            num_actions=num_actions,
            is_minerl=True
        )

        vae = VAE(input_shape=input_shape, latent_size=latent_size)
        vae.load_state_dict(torch.load(f"vae_models/minerl/{track_name}/encoder_{latent_size}_50.pth"))
        vae.to("cuda")
        vae.eval()
        freeze_parameters(vae)

        ssm_short = NextStepPredictionSSMMineRL(
            encoded_latent_size=latent_size,
            action_size=num_actions,
            hidden_state_size=hidden_state_size,
            context_size=ssm_context_length,
            is_minerl=True
        )
        ssm_short.load_state_dict(torch.load(f"ssm_models/ssm_{track_name}_250.pth"))
        ssm_short.to(device)
        freeze_parameters(ssm_short)

        ssm_long = NextStepPredictionSSMMineRL(
            encoded_latent_size=latent_size,
            action_size=num_actions,
            hidden_state_size=hidden_state_size,
            context_size=ssm_context_length,
            is_minerl=True
        )
        ssm_long.load_state_dict(torch.load(f"ssm_models/ssm_{track_name}_250.pth"))
        ssm_long.to(device)
        freeze_parameters(ssm_long)

        max_frames = None
        num_stored = 10
        full_file_paths = [os.path.join(traj_path, x) for x in os.listdir(traj_path)][:num_stored]
        for fp in full_file_paths:
            data = np.load(os.path.join(fp, "rendered.npz"), allow_pickle=True)
            action_keys = [x for x in data.files if x.startswith("action")]
            if "action$place" in action_keys:
                action_keys.remove("action$place")
            actions = []
            for x in action_keys:
                if data[x].ndim == 1:
                    actions.append(np.expand_dims(data[x].astype(np.float32), -1))
                else:
                    for i in range(data[x].shape[-1]):
                        actions.append(np.expand_dims(data[x][:, i].astype(np.float32), -1))
            observations = []
            actions = np.concatenate(actions, axis=-1)
            next_observations = []
            means = []
            logvars = []
            cap = cv2.VideoCapture(os.path.join(fp, "recording.mp4"))
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames and count >= max_frames):
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                latent, mean, logvar = vae.get_latent(torch.tensor(frame / 255.).float().unsqueeze(0).permute(0, 3, 1, 2).to(device))
                observations.append(latent)
                means.append(mean)
                logvars.append(logvar)

                count += 1
            cap.release()
            observations = torch.tensor(torch.cat(observations[:actions.shape[0]], dim=0))
            dones = torch.cat([torch.zeros(size=(actions.shape[0] - 1, 1)), torch.ones(size=(1, 1))], dim=0)
            actions = torch.tensor(actions)
            next_observations = torch.cat([observations[1:], torch.zeros(size=(1, *observations[0].size())).to(device)], dim=0)
            means = torch.tensor(torch.cat(means[:actions.shape[0]], dim=0))
            logvars = torch.tensor(torch.cat(logvars[:actions.shape[0]], dim=0))

            rollout_world_model.add_trajectory(
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                means=means,
                logvars=logvars
            )
            for l, a, l_next, m, lg in zip(observations, actions, next_observations, means, logvars):
                replay_world_model.add_transition(l, a, l_next, m, lg)

        full_file_paths_test = [os.path.join(traj_path, x) for x in os.listdir(traj_path)][num_stored:]
        test_traj = np.random.choice(full_file_paths_test, size=1)[0]

        data = np.load(os.path.join(test_traj, "rendered.npz"), allow_pickle=True)
        action_keys = [x for x in data.files if x.startswith("action")]
        if "action$place" in action_keys:
            action_keys.remove("action$place")
        actions = []
        for x in action_keys:
            if data[x].ndim == 1:
                actions.append(np.expand_dims(data[x].astype(np.float32), -1))
            else:
                for i in range(data[x].shape[-1]):
                    actions.append(np.expand_dims(data[x][:, i].astype(np.float32), -1))
        observations = []
        actions = np.concatenate(actions, axis=-1)
        next_observations = []
        means = []
        logvars = []
        cap = cv2.VideoCapture(os.path.join(test_traj, "recording.mp4"))
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and count >= max_frames):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            observations.append(frame)

            count += 1
        cap.release()
        observations = torch.tensor(np.stack(observations[:actions.shape[0]], axis=0))
        actions = torch.tensor(actions)
        next_observations = torch.cat([observations[1:], torch.zeros(size=(1, *observations[0].size()))], dim=0)
        test_traj = {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations
        }

        kl_divergence_with_base = {
            "rollout_one": [],
            "rollout_long": [],
            "replay_l2_one": [],
            "replay_l2_long": [],
            "replay_kl_one": [],
            "replay_kl_long": [],
            "ssm_one": [],
            "ssm_long": []
        }
        l1_distances_with_base = {
            "rollout_one": [],
            "rollout_long": [],
            "replay_l2_one": [],
            "replay_l2_long": [],
            "replay_kl_one": [],
            "replay_kl_long": [],
            "ssm_one": [],
            "ssm_long": []
        }
        ssim_with_base = {
            "rollout_one": [],
            "rollout_long": [],
            "replay_l2_one": [],
            "replay_l2_long": [],
            "replay_kl_one": [],
            "replay_kl_long": [],
            "ssm_one": [],
            "ssm_long": []
        }
        for s in range(num_samples):
            print(f"[{track_name}] Sample {s+1}/{num_samples}")
            tr_idx = np.random.randint(ssm_context_length, test_traj["observations"].shape[0] - horizon - 1)

            # Real
            real_seq = []
            real_mu_seq = []
            real_std_seq = []
            # Baseline 1-step
            ssm_short_term_seq = []
            ssm_short_stats_seq = []
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

            # DISCLAIMER: First step is done outside the for loop to retain indices for long-term search.

            # Real sequence
            real_seq.append(test_traj["observations"][tr_idx] / 255.)
            latent, mu, logvar = vae.get_latent(torch.tensor(test_traj["observations"][tr_idx] / 255.).float().unsqueeze(0).permute(0, 3, 1, 2).to(device))
            real_mu_seq.append(mu)
            real_std_seq.append(torch.sqrt(torch.exp(logvar)))
            # Rollout 1-step
            similar_transition, similar_indices_rollout = rollout_world_model.sample_similar(latent.cpu(), return_indices=True, action=test_traj["actions"][tr_idx] if condition_on_action else None)
            rollout_one_seq.append(similar_transition.observation)
            # Rollout long-term
            imagined_obs = []
            for ts, tr in enumerate(similar_indices_rollout.squeeze().numpy()):
                imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr]))
                if condition_on_action:
                    if bool((imagined_next.action == test_traj["actions"][tr_idx]).all()):
                        imagined_obs.append(imagined_next.observation)
                else:
                    imagined_obs.append(imagined_next.observation)
            # To avoid extreme estimations, we approximate without action conditioning whenever there are too few samples with the same action
            if len(imagined_obs) <= num_collected / 2:
                imagined_obs = []
                for ts, tr in enumerate(similar_indices_rollout.squeeze().numpy()):
                    imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr]))
                    imagined_obs.append(imagined_next.observation)
            rollout_long_seq.append(torch.cat(imagined_obs))
            # SSM 1-step
            ssm_context = test_traj["observations"][tr_idx - ssm_context_length:tr_idx]
            ssm_obs_buffer = [torch.tensor(x).permute(2, 0, 1).unsqueeze(0) for x in ssm_context]
            ssm_context, _, _ = vae.get_latent(torch.tensor(ssm_context / 255.).float().permute(0, 3, 1, 2).to(device))
            ssm_context_action = test_traj["actions"][tr_idx - 1]
            h_t = ssm_short.get_previous_hidden(ssm_context.flatten().unsqueeze(0), torch.tensor(ssm_context_action).view(1, -1).to(device), device=device)
            # Step one ahead
            ssm_obs_buffer.pop(0)
            ssm_obs_buffer.append(torch.tensor(test_traj["observations"][tr_idx]).permute(2, 0, 1).unsqueeze(0))
            o_t = torch.cat(ssm_obs_buffer, dim=0)
            o_t, _, _ = vae.get_latent((o_t / 255.).to(device))
            ssm_short_pred, stats_short = ssm_short.predict(h_t, o_t.unsqueeze(0), torch.tensor(test_traj["actions"][tr_idx]).view(1, 1, -1).to(device), device)
            context_obs, context_stats = ssm_short.get_context_prior(h_t, o_t.unsqueeze(0), torch.tensor(test_traj["actions"][tr_idx]).view(1, -1).to(device), device)
            ssm_short_term_seq.append(context_obs)
            ssm_short_stats_seq.append(context_stats)
            ssm_short_term_seq.append(ssm_short_pred[0])
            ssm_short_stats_seq.append(stats_short)
            # Replay L2 1-step
            similar_transition_replay, similar_indices_replay = replay_world_model.sample_similar(latent.cpu(), action=test_traj["actions"][tr_idx] if condition_on_action else None, k=10)
            obs = similar_transition_replay.observation
            replay_l2_one_seq.append(obs)
            # Replay L2 long-term
            replay_l2_long_seq.append(obs)
            # Replay KL 1-step
            similar_transition, similar_indices_kl, kl_div = replay_world_model.sample_similar_distribution(mu.cpu(), logvar.cpu(), action=test_traj["actions"][tr_idx] if condition_on_action else None)
            obs = torch.distributions.Normal(similar_transition.mean, similar_transition.sigma).rsample()
            replay_kl_one_seq.append(obs)
            replay_kl_div_one.append(kl_div.item())
            # Replay KL long-term
            replay_kl_long_seq.append(obs)
            replay_kl_div_long.append(kl_div.item())

            for i in range(1, horizon):
                # Base
                real_seq.append(test_traj["observations"][tr_idx + i] / 255.)
                latent, mu, logvar = vae.get_latent(torch.tensor(test_traj["observations"][tr_idx + i] / 255.).float().unsqueeze(0).permute(0, 3, 1, 2).to(device))
                real_mu_seq.append(mu)
                real_std_seq.append(torch.sqrt(torch.exp(logvar)))
                # One-step search-based
                similar_transition, _ = rollout_world_model.sample_similar(latent.cpu(), return_indices=False, action=test_traj["actions"][tr_idx + i] if condition_on_action else None)
                rollout_one_seq.append(similar_transition.next_observation)
                # Long-term search-based
                imagined_obs = []
                # Search best sample for each trajectory
                for ts, tr in enumerate(similar_indices_rollout.squeeze().numpy()):
                    imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr + i]))
                    if bool((imagined_next.action == test_traj["actions"][tr_idx]).all()):
                        imagined_obs.append(imagined_next.observation)
                # To avoid extreme estimations, we approximate without action conditioning whenever there are too few samples with the same action
                if len(imagined_obs) <= num_collected / 2:
                    imagined_obs = []
                    for ts, tr in enumerate(similar_indices_rollout.squeeze().numpy()):
                        imagined_next = rollout_world_model.get_transition(torch.tensor([ts]), torch.tensor([tr + i]))
                        imagined_obs.append(imagined_next.observation)
                rollout_long_seq.append(torch.cat(imagined_obs))
                # One-step SSM
                ssm_context = test_traj["observations"][tr_idx - ssm_context_length + i:tr_idx + i]
                ssm_obs_buffer = [torch.tensor(x).permute(2, 0, 1).unsqueeze(0) for x in ssm_context]
                ssm_context, _, _ = vae.get_latent(torch.tensor(ssm_context / 255.).float().permute(0, 3, 1, 2).to(device))
                ssm_context_action = test_traj["actions"][tr_idx - 1 + i]
                h_t = ssm_short.get_previous_hidden(ssm_context.flatten().unsqueeze(0), torch.tensor(ssm_context_action).view(1, -1).to(device), device=device)
                # Step one ahead
                ssm_obs_buffer.pop(0)
                ssm_obs_buffer.append(torch.tensor(test_traj["observations"][tr_idx + i]).permute(2, 0, 1).unsqueeze(0))
                o_t = torch.cat(ssm_obs_buffer, dim=0)
                o_t, _, _ = vae.get_latent((o_t / 255.).to(device))
                ssm_short_pred, stats_short = ssm_short.predict(h_t, o_t.unsqueeze(0), torch.tensor(test_traj["actions"][tr_idx + i]).view(1, 1, -1).to(device), device)
                ssm_short_term_seq.append(ssm_short_pred[0])
                ssm_short_stats_seq.append(stats_short)
                # Replay L2 1-step
                similar_transition, _ = replay_world_model.sample_similar(latent.cpu(), action=test_traj["actions"][tr_idx + i] if condition_on_action else None, k=10)
                obs = similar_transition.next_observation
                replay_l2_one_seq.append(obs)
                # Replay L2 long-term
                imagined_obs = []
                for tr in similar_indices_replay.squeeze().numpy():
                    imagined_next = replay_world_model.get_transition(torch.tensor([tr + i]))
                    imagined_obs.append(imagined_next.observation)
                replay_l2_long_seq.append(torch.cat(imagined_obs))
                # Replay KL 1-step
                _, next_mu, next_logvar = vae.get_latent(torch.tensor(test_traj["next_observations"][tr_idx + i] / 255.).float().unsqueeze(0).permute(0, 3, 1, 2).to(device))
                similar_transition, _, kl_div = replay_world_model.sample_similar_distribution(next_mu.cpu(), next_logvar.cpu(), action=test_traj["actions"][tr_idx + i] if condition_on_action else None)
                obs = torch.distributions.Normal(similar_transition.mean, similar_transition.sigma).rsample()
                replay_kl_one_seq.append(obs)
                replay_kl_div_one.append(kl_div.item())
                # Replay KL long-term
                imagined_next = replay_world_model.get_transition(torch.tensor([similar_indices_kl + i]))
                ref_distrib = torch.distributions.Normal(mu.cpu(), torch.sqrt(torch.exp(logvar)).cpu())
                next_distrib = torch.distributions.Normal(imagined_next.mean, imagined_next.sigma)
                obs_next = next_distrib.rsample()
                replay_kl_long_seq.append(obs_next.squeeze())
                replay_kl_div_long.append(torch.distributions.kl.kl_divergence(ref_distrib, next_distrib).sum(-1).item())

            # Adjust ssm sequence
            ssm_short_term_seq.pop(-1)
            ssm_short_stats_seq.pop(-1)

            # SSM long-term
            ssm_context = test_traj["observations"][tr_idx - ssm_context_length:tr_idx]
            ssm_obs_buffer = [torch.tensor(x).permute(2, 0, 1).unsqueeze(0) for x in ssm_context]
            ssm_context, _, _ = vae.get_latent(torch.tensor(ssm_context / 255.).float().permute(0, 3, 1, 2).to(device))
            ssm_context_action = test_traj["actions"][tr_idx - 1]
            h_t = ssm_long.get_previous_hidden(ssm_context.flatten().unsqueeze(0), torch.tensor(ssm_context_action).view(1, -1).to(device), device=device)
            # Step one ahead
            ssm_obs_buffer.pop(0)
            ssm_obs_buffer.append(torch.tensor(test_traj["observations"][tr_idx]).permute(2, 0, 1).unsqueeze(0))
            o_t = torch.cat(ssm_obs_buffer, dim=0)
            o_t, _, _ = vae.get_latent((o_t / 255.).to(device))
            ssm_long_term_seq, stats_long = ssm_long.predict(h_t, o_t.unsqueeze(0), torch.tensor(test_traj["actions"][tr_idx:tr_idx + horizon]).view(1, horizon, -1).to(device), device,
                                                             horizon=horizon)
            context_obs, context_stats = ssm_short.get_context_prior(h_t, o_t.unsqueeze(0), torch.tensor(test_traj["actions"][tr_idx]).view(1, -1).to(device), device)
            ssm_long_term_seq.pop(-1)
            ssm_long_term_seq.insert(0, context_obs)
            stats_long["posterior_mean"].pop(-1)
            stats_long["posterior_logvar"].pop(-1)
            stats_long["posterior_mean"].insert(0, context_stats["posterior_mean"].squeeze(0))
            stats_long["posterior_logvar"].insert(0, context_stats["posterior_logvar"].squeeze(0))

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
            # One-step SSM
            decoded_ssm_short = vae.decode(torch.cat(ssm_short_term_seq).to(device))
            dist_ssm_short = torch.distributions.normal.Normal(torch.cat([x["posterior_mean"][0] for x in ssm_short_stats_seq], dim=0), torch.cat([torch.sqrt(torch.exp(x["posterior_logvar"][0])) for x in ssm_short_stats_seq]))
            kl_divergence_with_base["ssm_one"].append(torch.distributions.kl.kl_divergence(dist_base, dist_ssm_short).sum(-1).unsqueeze(0))
            l1_distances_with_base["ssm_one"].append(torch.nn.functional.l1_loss(decoded_ssm_short, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["ssm_one"].append(
                torch.tensor([ssim(decoded_ssm_short[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in range(horizon)]).unsqueeze(0))
            # Long-term SSM
            decoded_ssm_long = vae.decode(torch.cat(ssm_long_term_seq).to(device))
            dist_ssm_long = torch.distributions.normal.Normal(torch.cat(stats_long["posterior_mean"], dim=0), torch.sqrt(torch.exp(torch.cat(stats_long["posterior_logvar"], dim=0))))
            kl_divergence_with_base["ssm_long"].append(torch.distributions.kl.kl_divergence(dist_base, dist_ssm_long).sum(-1).unsqueeze(0))
            l1_distances_with_base["ssm_long"].append(torch.nn.functional.l1_loss(decoded_ssm_long, base_seq_tensor, reduction='none').mean(dim=[1, 2, 3]).unsqueeze(0))
            ssim_with_base["ssm_long"].append(
                torch.tensor([ssim(decoded_ssm_long[i].cpu().permute(1, 2, 0).numpy(), base_seq_tensor[i].cpu().permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in range(horizon)]).unsqueeze(0))
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
            # Save samples
            np.savez_compressed(
                f"experiments/prediction_horizon/minerl/{track_name}/samples/one_step/decoded_one_step_sample_{s}.npz",
                real=np.array([cv2.resize(x.cpu().numpy(), (128, 128)) for x in real_seq]),
                ssm=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_ssm_short]),
                rollout=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_rollout_one]),
                replay_l2=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_l2_one]),
                replay_kl=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_kl_one])
            )

            np.savez_compressed(
                f"experiments/prediction_horizon/minerl/{track_name}/samples/long_term/decoded_long_term_sample_{s}.npz",
                real=np.array([cv2.resize(x.cpu().numpy(), (128, 128)) for x in real_seq]),
                ssm=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_ssm_long]),
                rollout=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_rollout_long]),
                replay_l2=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_l2_long]),
                replay_kl=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_kl_long])
            )

        # Stat plots
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
        # Rollout one-step
        mean_kl_dist = torch.cat(kl_divergence_with_base["rollout_one"], dim=0).mean(dim=0).cpu().numpy()
        std_kl_dist = torch.cat(kl_divergence_with_base["rollout_one"], dim=0).std(dim=0).cpu().numpy()
        mean_l1_dist = torch.cat(l1_distances_with_base["rollout_one"], dim=0).mean(dim=0).cpu().numpy()
        std_l1_dist = torch.cat(l1_distances_with_base["rollout_one"], dim=0).std(dim=0).cpu().numpy()
        mean_ssim_dist = torch.cat(ssim_with_base["rollout_one"], dim=0).mean(dim=0).cpu().numpy()
        std_ssim_dist = torch.cat(ssim_with_base["rollout_one"], dim=0).std(dim=0).cpu().numpy()
        # Save overall results
        overall_results_kl["rollout_one"]["mean"] += mean_kl_dist
        overall_results_kl["rollout_one"]["std"] += std_kl_dist
        overall_results_l1["rollout_one"]["mean"] += mean_l1_dist
        overall_results_l1["rollout_one"]["std"] += std_l1_dist
        overall_results_ssim["rollout_one"]["mean"] += mean_ssim_dist
        overall_results_ssim["rollout_one"]["std"] += std_ssim_dist
        # KL
        rollout_one, = ax[0, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:red')
        ax[0, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:red')
        # L1
        ax[0, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:red')
        ax[0, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:red')
        # SSIM
        ax[0, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:red')
        ax[0, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:red')
        # Search long-term
        mean_kl_dist = torch.cat(kl_divergence_with_base["rollout_long"], dim=0).mean(dim=0).cpu().numpy()
        std_kl_dist = torch.cat(kl_divergence_with_base["rollout_long"], dim=0).std(dim=0).cpu().numpy()
        mean_l1_dist = torch.cat(l1_distances_with_base["rollout_long"], dim=0).mean(dim=0).cpu().numpy()
        std_l1_dist = torch.cat(l1_distances_with_base["rollout_long"], dim=0).std(dim=0).cpu().numpy()
        mean_ssim_dist = torch.cat(ssim_with_base["rollout_long"], dim=0).mean(dim=0).cpu().numpy()
        std_ssim_dist = torch.cat(ssim_with_base["rollout_long"], dim=0).std(dim=0).cpu().numpy()
        print(f"Track: {track_name} | Rollout | KL = {np.mean(mean_kl_dist)} - MSE = {np.mean(mean_l1_dist)} - SSIM = {np.mean(mean_ssim_dist)}")
        # Save overall results
        overall_results_kl["rollout_long"]["mean"] += mean_kl_dist
        overall_results_kl["rollout_long"]["std"] += std_kl_dist
        overall_results_l1["rollout_long"]["mean"] += mean_l1_dist
        overall_results_l1["rollout_long"]["std"] += std_l1_dist
        overall_results_ssim["rollout_long"]["mean"] += mean_ssim_dist
        overall_results_ssim["rollout_long"]["std"] += std_ssim_dist
        # KL
        rollout_long, = ax[1, 0].plot(mean_kl_dist, label="Search-based (long-term)", c='tab:red')
        ax[1, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:red')
        # L1
        ax[1, 1].plot(mean_l1_dist, label="Search-based (long-term)", c='tab:red')
        ax[1, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:red')
        # SSIM
        ax[1, 2].plot(mean_ssim_dist, label="Search-based (long-term)", c='tab:red')
        ax[1, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:red')
        # SSM one-step
        mean_kl_dist = torch.cat(kl_divergence_with_base["ssm_one"], dim=0).mean(dim=0).cpu().numpy()
        std_kl_dist = torch.cat(kl_divergence_with_base["ssm_one"], dim=0).std(dim=0).cpu().numpy()
        mean_l1_dist = torch.cat(l1_distances_with_base["ssm_one"], dim=0).mean(dim=0).cpu().numpy()
        std_l1_dist = torch.cat(l1_distances_with_base["ssm_one"], dim=0).std(dim=0).cpu().numpy()
        mean_ssim_dist = torch.cat(ssim_with_base["ssm_one"], dim=0).mean(dim=0).cpu().numpy()
        std_ssim_dist = torch.cat(ssim_with_base["ssm_one"], dim=0).std(dim=0).cpu().numpy()
        # Save overall results
        overall_results_kl["ssm_one"]["mean"] += mean_kl_dist
        overall_results_kl["ssm_one"]["std"] += std_kl_dist
        overall_results_l1["ssm_one"]["mean"] += mean_l1_dist
        overall_results_l1["ssm_one"]["std"] += std_l1_dist
        overall_results_ssim["ssm_one"]["mean"] += mean_ssim_dist
        overall_results_ssim["ssm_one"]["std"] += std_ssim_dist
        # KL
        ssm_one, = ax[0, 0].plot(torch.cat(kl_divergence_with_base["ssm_one"], dim=0).mean(dim=0).cpu().numpy(), label="SSM (1-step)", c='tab:blue')
        ax[0, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:blue')
        # L1
        ax[0, 1].plot(mean_l1_dist, label="SSM (1-step", c='tab:blue')
        ax[0, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:blue')
        # SSIM
        ax[0, 2].plot(mean_ssim_dist, label="SSM (1-step)", c='tab:blue')
        ax[0, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:blue')
        # SSM long-term
        mean_kl_dist = torch.cat(kl_divergence_with_base["ssm_long"], dim=0).mean(dim=0).cpu().numpy()
        std_kl_dist = torch.cat(kl_divergence_with_base["ssm_long"], dim=0).std(dim=0).cpu().numpy()
        mean_l1_dist = torch.cat(l1_distances_with_base["ssm_long"], dim=0).mean(dim=0).cpu().numpy()
        std_l1_dist = torch.cat(l1_distances_with_base["ssm_long"], dim=0).std(dim=0).cpu().numpy()
        mean_ssim_dist = torch.cat(ssim_with_base["ssm_long"], dim=0).mean(dim=0).cpu().numpy()
        std_ssim_dist = torch.cat(ssim_with_base["ssm_long"], dim=0).std(dim=0).cpu().numpy()
        print(f"Track: {track_name} | SSM | KL = {np.mean(mean_kl_dist)} - MSE = {np.mean(mean_l1_dist)} - SSIM = {np.mean(mean_ssim_dist)}")
        # Save overall results
        overall_results_kl["ssm_long"]["mean"] += mean_kl_dist
        overall_results_kl["ssm_long"]["std"] += std_kl_dist
        overall_results_l1["ssm_long"]["mean"] += mean_l1_dist
        overall_results_l1["ssm_long"]["std"] += std_l1_dist
        overall_results_ssim["ssm_long"]["mean"] += mean_ssim_dist
        overall_results_ssim["ssm_long"]["std"] += std_ssim_dist
        # KL
        ssm_long_fig, = ax[1, 0].plot(torch.cat(kl_divergence_with_base["ssm_long"], dim=0).mean(dim=0).cpu().numpy(), label="SSM (long)", c='tab:blue')
        ax[1, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:blue')
        # L1
        ax[1, 1].plot(mean_l1_dist, label="SSM (long-term)", c='tab:blue')
        ax[1, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:blue')
        # SSIM
        ax[1, 2].plot(mean_ssim_dist, label="SSM (long-term)", c='tab:blue')
        ax[1, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:blue')
        # Replay 1-step
        mean_kl_dist = torch.cat(kl_divergence_with_base["replay_l2_one"], dim=0).mean(dim=0).cpu().numpy()
        std_kl_dist = torch.cat(kl_divergence_with_base["replay_l2_one"], dim=0).std(dim=0).cpu().numpy()
        mean_l1_dist = torch.cat(l1_distances_with_base["replay_l2_one"], dim=0).mean(dim=0).cpu().numpy()
        std_l1_dist = torch.cat(l1_distances_with_base["replay_l2_one"], dim=0).std(dim=0).cpu().numpy()
        mean_ssim_dist = torch.cat(ssim_with_base["replay_l2_one"], dim=0).mean(dim=0).cpu().numpy()
        std_ssim_dist = torch.cat(ssim_with_base["replay_l2_one"], dim=0).std(dim=0).cpu().numpy()
        # Save overall results
        overall_results_kl["replay_l2_one"]["mean"] += mean_kl_dist
        overall_results_kl["replay_l2_one"]["std"] += std_kl_dist
        overall_results_l1["replay_l2_one"]["mean"] += mean_l1_dist
        overall_results_l1["replay_l2_one"]["std"] += std_l1_dist
        overall_results_ssim["replay_l2_one"]["mean"] += mean_ssim_dist
        overall_results_ssim["replay_l2_one"]["std"] += std_ssim_dist
        # KL
        replay_l2_one, = ax[0, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:orange')
        ax[0, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:orange')
        # L1
        ax[0, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:orange')
        ax[0, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:orange')
        # SSIM
        ax[0, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:orange')
        ax[0, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:orange')
        # Replay L2 long-term
        mean_kl_dist = torch.cat(kl_divergence_with_base["replay_l2_long"], dim=0).mean(dim=0).cpu().numpy()
        std_kl_dist = torch.cat(kl_divergence_with_base["replay_l2_long"], dim=0).std(dim=0).cpu().numpy()
        mean_l1_dist = torch.cat(l1_distances_with_base["replay_l2_long"], dim=0).mean(dim=0).cpu().numpy()
        std_l1_dist = torch.cat(l1_distances_with_base["replay_l2_long"], dim=0).std(dim=0).cpu().numpy()
        mean_ssim_dist = torch.cat(ssim_with_base["replay_l2_long"], dim=0).mean(dim=0).cpu().numpy()
        std_ssim_dist = torch.cat(ssim_with_base["replay_l2_long"], dim=0).std(dim=0).cpu().numpy()
        print(f"Track: {track_name} | Replay-L2 | KL = {np.mean(mean_kl_dist)} - MSE = {np.mean(mean_l1_dist)} - SSIM = {np.mean(mean_ssim_dist)}")
        # Save overall results
        overall_results_kl["replay_l2_long"]["mean"] += mean_kl_dist
        overall_results_kl["replay_l2_long"]["std"] += std_kl_dist
        overall_results_l1["replay_l2_long"]["mean"] += mean_l1_dist
        overall_results_l1["replay_l2_long"]["std"] += std_l1_dist
        overall_results_ssim["replay_l2_long"]["mean"] += mean_ssim_dist
        overall_results_ssim["replay_l2_long"]["std"] += std_ssim_dist
        # KL
        replay_l2_long, = ax[1, 0].plot(mean_kl_dist, label="Search-based (long-term)", c='tab:orange')
        ax[1, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:orange')
        # L1
        ax[1, 1].plot(mean_l1_dist, label="Search-based (long-term)", c='tab:orange')
        ax[1, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:orange')
        # SSIM
        ax[1, 2].plot(mean_ssim_dist, label="Search-based (long-term)", c='tab:orange')
        ax[1, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:orange')
        # Replay KL 1-step
        mean_kl_dist = torch.cat(kl_divergence_with_base["replay_kl_one"], dim=0).mean(dim=0).cpu().numpy()
        std_kl_dist = torch.cat(kl_divergence_with_base["replay_kl_one"], dim=0).std(dim=0).cpu().numpy()
        mean_l1_dist = torch.cat(l1_distances_with_base["replay_kl_one"], dim=0).mean(dim=0).cpu().numpy()
        std_l1_dist = torch.cat(l1_distances_with_base["replay_kl_one"], dim=0).std(dim=0).cpu().numpy()
        mean_ssim_dist = torch.cat(ssim_with_base["replay_kl_one"], dim=0).mean(dim=0).cpu().numpy()
        std_ssim_dist = torch.cat(ssim_with_base["replay_kl_one"], dim=0).std(dim=0).cpu().numpy()
        # Save overall results
        overall_results_kl["replay_kl_one"]["mean"] += mean_kl_dist
        overall_results_kl["replay_kl_one"]["std"] += std_kl_dist
        overall_results_l1["replay_kl_one"]["mean"] += mean_l1_dist
        overall_results_l1["replay_kl_one"]["std"] += std_l1_dist
        overall_results_ssim["replay_kl_one"]["mean"] += mean_ssim_dist
        overall_results_ssim["replay_kl_one"]["std"] += std_ssim_dist
        # KL
        replay_kl_one, = ax[0, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:brown')
        ax[0, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:brown')
        # L1
        ax[0, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:brown')
        ax[0, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:brown')
        # SSIM
        ax[0, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:brown')
        ax[0, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:brown')
        # Replay KL long-term
        mean_kl_dist = torch.cat(kl_divergence_with_base["replay_kl_long"], dim=0).mean(dim=0).cpu().numpy()
        std_kl_dist = torch.cat(kl_divergence_with_base["replay_kl_long"], dim=0).std(dim=0).cpu().numpy()
        mean_l1_dist = torch.cat(l1_distances_with_base["replay_kl_long"], dim=0).mean(dim=0).cpu().numpy()
        std_l1_dist = torch.cat(l1_distances_with_base["replay_kl_long"], dim=0).std(dim=0).cpu().numpy()
        mean_ssim_dist = torch.cat(ssim_with_base["replay_kl_long"], dim=0).mean(dim=0).cpu().numpy()
        std_ssim_dist = torch.cat(ssim_with_base["replay_kl_long"], dim=0).std(dim=0).cpu().numpy()
        print(f"Track: {track_name} | Replay-KL | KL = {np.mean(mean_kl_dist)} - MSE = {np.mean(mean_l1_dist)} - SSIM = {np.mean(mean_ssim_dist)}")
        # Save overall results
        overall_results_kl["replay_kl_long"]["mean"] += mean_kl_dist
        overall_results_kl["replay_kl_long"]["std"] += std_kl_dist
        overall_results_l1["replay_kl_long"]["mean"] += mean_l1_dist
        overall_results_l1["replay_kl_long"]["std"] += std_l1_dist
        overall_results_ssim["replay_kl_long"]["mean"] += mean_ssim_dist
        overall_results_ssim["replay_kl_long"]["std"] += std_ssim_dist
        # KL
        replay_kl_long, = ax[1, 0].plot(mean_kl_dist, label="Search-based (long-term)", c='tab:brown')
        ax[1, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:brown')
        # L1
        ax[1, 1].plot(mean_l1_dist, label="Search-based (long-term)", c='tab:brown')
        ax[1, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:brown')
        # SSIM
        ax[1, 2].plot(mean_ssim_dist, label="Search-based (long-term)", c='tab:brown')
        ax[1, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:brown')

        # General
        ax[0, 0].set_ylabel("KL Divergence", fontsize=20)
        ax[0, 1].set_ylabel("L1 Distance", fontsize=20)
        ax[0, 2].set_ylabel("SSIM", fontsize=20)
        ax[1, 0].set_ylabel("KL Divergence", fontsize=20)
        ax[1, 1].set_ylabel("L1 Distance", fontsize=20)
        ax[1, 2].set_ylabel("SSIM", fontsize=20)
        ax[1, 1].set_xlabel("Timestep", fontsize=20)

        ax[0, 0].tick_params(axis='both', which='major', labelsize=16)
        ax[0, 1].tick_params(axis='both', which='major', labelsize=16)
        ax[0, 2].tick_params(axis='both', which='major', labelsize=16)
        ax[1, 0].tick_params(axis='both', which='major', labelsize=16)
        ax[1, 1].tick_params(axis='both', which='major', labelsize=16)
        ax[1, 2].tick_params(axis='both', which='major', labelsize=16)

        ax[0, 0].set_xticks([0, 4, 8, 12, 16, 20])
        ax[0, 1].set_xticks([0, 4, 8, 12, 16, 20])
        ax[0, 2].set_xticks([0, 4, 8, 12, 16, 20])
        ax[1, 0].set_xticks([0, 4, 8, 12, 16, 20])
        ax[1, 1].set_xticks([0, 4, 8, 12, 16, 20])
        ax[1, 2].set_xticks([0, 4, 8, 12, 16, 20])

        ax[0, 1].set_yticks([0, 0.1, 0.2, 0.3])
        ax[1, 1].set_yticks([0, 0.1, 0.2, 0.3])
        ax[0, 2].set_yticks([0.0, 0.2, 0.4])
        ax[1, 2].set_yticks([0.0, 0.2, 0.4])

        ax[0, 0].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)
        ax[0, 1].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)
        ax[0, 2].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)
        ax[1, 0].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)
        ax[1, 1].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)
        ax[1, 2].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)

        ax[0, 1].set_yticklabels(["0", ".1", ".2", ".3"])
        ax[1, 1].set_yticklabels(["0", ".1", ".2", ".3"])
        ax[0, 2].set_yticklabels([0.0, 0.2, 0.4])
        ax[1, 2].set_yticklabels([0.0, 0.2, 0.4])

        fig.legend([ssm_one, rollout_one, replay_l2_one, replay_kl_one], ["Baseline", "Rollout", "Replay-L2", "Replay-KL"], loc="upper right", fontsize=16)
        fig.savefig(f"experiments/prediction_horizon/minerl/{track_name}/numerical_comparison.png")
        fig.clear()

    # Benchmark results
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
    for k in overall_results_kl.keys():
        overall_results_kl[k]["mean"] = overall_results_kl[k]["mean"] / float(len(env_names))
        overall_results_kl[k]["std"] = overall_results_kl[k]["std"] / float(len(env_names))
        overall_results_l1[k]["mean"] = overall_results_l1[k]["mean"] / float(len(env_names))
        overall_results_l1[k]["std"] = overall_results_l1[k]["std"] / float(len(env_names))
        overall_results_ssim[k]["mean"] = overall_results_ssim[k]["mean"] / float(len(env_names))
        overall_results_ssim[k]["std"] = overall_results_ssim[k]["std"] / float(len(env_names))

    # Rollout 1-step
    mean_kl_dist = overall_results_kl["rollout_one"]["mean"]
    std_kl_dist = overall_results_kl["rollout_one"]["std"]
    mean_l1_dist = overall_results_l1["rollout_one"]["mean"]
    std_l1_dist = overall_results_l1["rollout_one"]["std"]
    mean_ssim_dist = overall_results_ssim["rollout_one"]["mean"]
    std_ssim_dist = overall_results_ssim["rollout_one"]["std"]
    rollout_one, = ax[0, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:red')
    ax[0, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:red')
    # L1
    ax[0, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:red')
    ax[0, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:red')
    # SSIM
    ax[0, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:red')
    ax[0, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:red')

    # Rollout long-term
    mean_kl_dist = overall_results_kl["rollout_long"]["mean"]
    std_kl_dist = overall_results_kl["rollout_long"]["std"]
    mean_l1_dist = overall_results_l1["rollout_long"]["mean"]
    std_l1_dist = overall_results_l1["rollout_long"]["std"]
    mean_ssim_dist = overall_results_ssim["rollout_long"]["mean"]
    std_ssim_dist = overall_results_ssim["rollout_long"]["std"]
    rollout_long, = ax[1, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:red')
    ax[1, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:red')
    # L1
    ax[1, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:red')
    ax[1, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:red')
    # SSIM
    ax[1, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:red')
    ax[1, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:red')

    # SSM 1-step
    mean_kl_dist = overall_results_kl["ssm_one"]["mean"]
    std_kl_dist = overall_results_kl["ssm_one"]["std"]
    mean_l1_dist = overall_results_l1["ssm_one"]["mean"]
    std_l1_dist = overall_results_l1["ssm_one"]["std"]
    mean_ssim_dist = overall_results_ssim["ssm_one"]["mean"]
    std_ssim_dist = overall_results_ssim["ssm_one"]["std"]
    ssm_one, = ax[0, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:blue')
    ax[0, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:blue')
    # L1
    ax[0, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:blue')
    ax[0, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:blue')
    # SSIM
    ax[0, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:blue')
    ax[0, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:blue')

    # SSM long-term
    mean_kl_dist = overall_results_kl["ssm_long"]["mean"]
    std_kl_dist = overall_results_kl["ssm_long"]["std"]
    mean_l1_dist = overall_results_l1["ssm_long"]["mean"]
    std_l1_dist = overall_results_l1["ssm_long"]["std"]
    mean_ssim_dist = overall_results_ssim["ssm_long"]["mean"]
    std_ssim_dist = overall_results_ssim["ssm_long"]["std"]
    ssm_long_fig, = ax[1, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:blue')
    ax[1, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:blue')
    # L1
    ax[1, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:blue')
    ax[1, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:blue')
    # SSIM
    ax[1, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:blue')
    ax[1, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:blue')

    # Replay L2 1-step
    mean_kl_dist = overall_results_kl["replay_l2_one"]["mean"]
    std_kl_dist = overall_results_kl["replay_l2_one"]["std"]
    mean_l1_dist = overall_results_l1["replay_l2_one"]["mean"]
    std_l1_dist = overall_results_l1["replay_l2_one"]["std"]
    mean_ssim_dist = overall_results_ssim["replay_l2_one"]["mean"]
    std_ssim_dist = overall_results_ssim["replay_l2_one"]["std"]
    replay_l2_one, = ax[0, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:orange')
    ax[0, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:orange')
    # L1
    ax[0, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:orange')
    ax[0, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:orange')
    # SSIM
    ax[0, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:orange')
    ax[0, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:orange')

    # Replay L2 long-term
    mean_kl_dist = overall_results_kl["replay_l2_long"]["mean"]
    std_kl_dist = overall_results_kl["replay_l2_long"]["std"]
    mean_l1_dist = overall_results_l1["replay_l2_long"]["mean"]
    std_l1_dist = overall_results_l1["replay_l2_long"]["std"]
    mean_ssim_dist = overall_results_ssim["replay_l2_long"]["mean"]
    std_ssim_dist = overall_results_ssim["replay_l2_long"]["std"]
    replay_l2_long, = ax[1, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:orange')
    ax[1, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:orange')
    # L1
    ax[1, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:orange')
    ax[1, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:orange')
    # SSIM
    ax[1, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:orange')
    ax[1, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:orange')

    # Replay KL 1-step
    mean_kl_dist = overall_results_kl["replay_kl_one"]["mean"]
    std_kl_dist = overall_results_kl["replay_kl_one"]["std"]
    mean_l1_dist = overall_results_l1["replay_kl_one"]["mean"]
    std_l1_dist = overall_results_l1["replay_kl_one"]["std"]
    mean_ssim_dist = overall_results_ssim["replay_kl_one"]["mean"]
    std_ssim_dist = overall_results_ssim["replay_kl_one"]["std"]
    replay_kl_one, = ax[0, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:brown')
    ax[0, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:brown')
    # L1
    ax[0, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:brown')
    ax[0, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:brown')
    # SSIM
    ax[0, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:brown')
    ax[0, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:brown')

    # SSM long-term
    mean_kl_dist = overall_results_kl["replay_kl_long"]["mean"]
    std_kl_dist = overall_results_kl["replay_kl_long"]["std"]
    mean_l1_dist = overall_results_l1["replay_kl_long"]["mean"]
    std_l1_dist = overall_results_l1["replay_kl_long"]["std"]
    mean_ssim_dist = overall_results_ssim["replay_kl_long"]["mean"]
    std_ssim_dist = overall_results_ssim["replay_kl_long"]["std"]
    replay_kl_long, = ax[1, 0].plot(mean_kl_dist, label="Search-based (1-step)", c='tab:brown')
    ax[1, 0].fill_between(np.arange(mean_kl_dist.shape[0]), mean_kl_dist - std_kl_dist, mean_kl_dist + std_kl_dist, alpha=0.2, color='tab:brown')
    # L1
    ax[1, 1].plot(mean_l1_dist, label="Search-based (1-step)", c='tab:brown')
    ax[1, 1].fill_between(np.arange(mean_l1_dist.shape[0]), mean_l1_dist - std_l1_dist, mean_l1_dist + std_l1_dist, alpha=0.2, color='tab:brown')
    # SSIM
    ax[1, 2].plot(mean_ssim_dist, label="Search-based (1-step)", c='tab:brown')
    ax[1, 2].fill_between(np.arange(mean_ssim_dist.shape[0]), mean_ssim_dist - std_ssim_dist, mean_ssim_dist + std_ssim_dist, alpha=0.2, color='tab:brown')

    ax[0, 0].set_ylabel("KL Divergence", fontsize=20)
    ax[0, 1].set_ylabel("L1 Distance", fontsize=20)
    ax[0, 2].set_ylabel("SSIM", fontsize=20)
    ax[1, 0].set_ylabel("KL Divergence", fontsize=20)
    ax[1, 1].set_ylabel("L1 Distance", fontsize=20)
    ax[1, 2].set_ylabel("SSIM", fontsize=20)
    ax[1, 1].set_xlabel("Timestep", fontsize=20)

    ax[0, 0].tick_params(axis='both', which='major', labelsize=16)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=16)
    ax[0, 2].tick_params(axis='both', which='major', labelsize=16)
    ax[1, 0].tick_params(axis='both', which='major', labelsize=16)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=16)
    ax[1, 2].tick_params(axis='both', which='major', labelsize=16)

    ax[0, 0].set_xticks([0, 4, 8, 12, 16, 20])
    ax[0, 1].set_xticks([0, 4, 8, 12, 16, 20])
    ax[0, 2].set_xticks([0, 4, 8, 12, 16, 20])
    ax[1, 0].set_xticks([0, 4, 8, 12, 16, 20])
    ax[1, 1].set_xticks([0, 4, 8, 12, 16, 20])
    ax[1, 2].set_xticks([0, 4, 8, 12, 16, 20])

    ax[0, 1].set_yticks([0, 0.1, 0.2, 0.3])
    ax[1, 1].set_yticks([0, 0.1, 0.2, 0.3])
    ax[0, 2].set_yticks([0.0, 0.2, 0.4])
    ax[1, 2].set_yticks([0.0, 0.2, 0.4])

    ax[0, 0].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)
    ax[0, 1].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)
    ax[0, 2].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)
    ax[1, 0].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)
    ax[1, 1].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)
    ax[1, 2].set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=16)

    ax[0, 1].set_yticklabels(["0", ".1", ".2", ".3"])
    ax[1, 1].set_yticklabels(["0", ".1", ".2", ".3"])
    ax[0, 2].set_yticklabels([0.0, 0.2, 0.4])
    ax[1, 2].set_yticklabels([0.0, 0.2, 0.4])

    fig.legend([ssm_one, rollout_one, replay_l2_one, replay_kl_one], ["Baseline", "Rollout", "Replay-L2", "Replay-KL"], loc="upper right", fontsize=16)
    fig.savefig(f"experiments/prediction_horizon/minerl/benchmark_numerical_comparison.png")
    fig.clear()
