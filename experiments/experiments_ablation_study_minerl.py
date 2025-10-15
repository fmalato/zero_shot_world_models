import os
import time

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
    num_actions = 10
    input_shape = (1, 3, 64, 64)
    latent_size = 512
    hidden_state_size = 256
    max_episode_len = 6000
    gamma = 0.99
    context_length = 1
    ssm_context_length = 4
    horizon = 20
    data_path = "minerl"
    track_name = "MineRLTreechop-v0"
    device = "cpu"
    condition_on_action = True
    save_samples = True

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

    num_trajectories = [10, 20, 40, 80]

    traj_path = f"{data_path}/{track_name}"
    test_traj_path = f"{data_path}/{track_name}"
    heatmap_kl_one_step = np.zeros(shape=(4, len(num_trajectories)))
    heatmap_kl_long_term = np.zeros(shape=(4, len(num_trajectories)))
    heatmap_l1_one_step = np.zeros(shape=(4, len(num_trajectories)))
    heatmap_l1_long_term = np.zeros(shape=(4, len(num_trajectories)))
    heatmap_ssim_one_step = np.zeros(shape=(4, len(num_trajectories)))
    heatmap_ssim_long_term = np.zeros(shape=(4, len(num_trajectories)))

    wallclock_time = {
        "rollout_one": {},
        "rollout_long": {},
        "replay_l2_one": {},
        "replay_l2_long": {},
        "replay_kl_one": {},
        "replay_kl_long": {},
        "ssm_one": {},
        "ssm_long": {}
    }

    for traj_idx, num_collected in enumerate(num_trajectories):
        for k in wallclock_time.keys():
            wallclock_time[k][num_collected] = []

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
        vae.to(device)
        vae.eval()
        freeze_parameters(vae)

        ssm_short = NextStepPredictionSSMMineRL(
            encoded_latent_size=latent_size,
            action_size=num_actions,
            hidden_state_size=hidden_state_size,
            context_size=ssm_context_length,
            is_minerl=True
        )
        ssm_short.load_state_dict(torch.load(f"ssm_models/ssm_ablation/minerl/ssm_{track_name}_250_{num_collected}.pth"))
        ssm_short.to(device)
        freeze_parameters(ssm_short)

        ssm_long = NextStepPredictionSSMMineRL(
            encoded_latent_size=latent_size,
            action_size=num_actions,
            hidden_state_size=hidden_state_size,
            context_size=ssm_context_length,
            is_minerl=True
        )
        ssm_long.load_state_dict(torch.load(f"ssm_models/ssm_ablation/minerl/ssm_{track_name}_250_{num_collected}.pth"))
        ssm_long.to(device)
        freeze_parameters(ssm_long)

        max_frames = None
        full_file_paths = [os.path.join(traj_path, x) for x in os.listdir(traj_path)][:num_collected]
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
            observations = torch.cat(observations[:actions.shape[0]], dim=0)
            dones = torch.cat([torch.zeros(size=(actions.shape[0] - 1, 1)), torch.ones(size=(1, 1))], dim=0)
            actions = torch.tensor(actions)
            next_observations = torch.cat([observations[1:], torch.zeros(size=(1, *observations[0].size())).to(device)], dim=0)
            means = torch.cat(means[:actions.shape[0]], dim=0)
            logvars = torch.cat(logvars[:actions.shape[0]], dim=0)

            rollout_world_model.add_trajectory(
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                means=means,
                logvars=logvars
            )
            for l, a, l_next, m, lg in zip(observations, actions, next_observations, means, logvars):
                replay_world_model.add_transition(l, a, l_next, m, lg)

        full_file_paths_test = [os.path.join(traj_path, x) for x in os.listdir(traj_path)][num_collected:]
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
            print(f"[{track_name} | Collected: {num_collected}] Sample {s+1}/{num_samples}")
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
            latent, mu, logvar = vae.get_latent((test_traj["observations"][tr_idx] / 255.).float().unsqueeze(0).permute(0, 3, 1, 2).to(device))
            real_mu_seq.append(mu)
            real_std_seq.append(torch.sqrt(torch.exp(logvar)))
            # Rollout 1-step
            start = time.time()
            similar_transition, similar_indices_rollout = rollout_world_model.sample_similar(latent.cpu(), return_indices=True, action=test_traj["actions"][tr_idx])
            end = time.time()
            wallclock_time["rollout_one"][num_collected].append(end - start)
            rollout_one_seq.append(similar_transition.observation)
            # Rollout long-term
            imagined_obs = []
            start = time.time()
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
            wallclock_time["rollout_long"][num_collected].append(time.time() - start)
            rollout_long_seq.append(torch.cat(imagined_obs))
            # SSM 1-step
            ssm_context = test_traj["observations"][tr_idx - ssm_context_length:tr_idx]
            ssm_obs_buffer = [x.permute(2, 0, 1).unsqueeze(0) for x in ssm_context]
            ssm_context, _, _ = vae.get_latent((ssm_context / 255.).float().permute(0, 3, 1, 2).to(device))
            ssm_context_action = test_traj["actions"][tr_idx - 1]
            start = time.time()
            h_t = ssm_short.get_previous_hidden(ssm_context.flatten().unsqueeze(0), ssm_context_action.view(1, -1).to(device), device=device)
            # Step one ahead
            ssm_obs_buffer.pop(0)
            ssm_obs_buffer.append(test_traj["observations"][tr_idx].permute(2, 0, 1).unsqueeze(0))
            o_t = torch.cat(ssm_obs_buffer, dim=0)
            o_t, _, _ = vae.get_latent((o_t / 255.).to(device))
            ssm_short_pred, stats_short = ssm_short.predict(h_t, o_t.unsqueeze(0), test_traj["actions"][tr_idx].view(1, 1, -1).to(device), device)
            context_obs, context_stats = ssm_short.get_context_prior(h_t, o_t.unsqueeze(0), test_traj["actions"][tr_idx].view(1, -1).to(device), device)
            wallclock_time["ssm_one"][num_collected].append(time.time() - start)
            ssm_short_term_seq.append(context_obs)
            ssm_short_stats_seq.append(context_stats)
            ssm_short_term_seq.append(ssm_short_pred[0])
            ssm_short_stats_seq.append(stats_short)
            # Replay L2 1-step
            start = time.time()
            similar_transition_replay, similar_indices_replay = replay_world_model.sample_similar(latent.cpu(), action=test_traj["actions"][tr_idx], k=10)
            end = time.time()
            wallclock_time["replay_l2_one"][num_collected].append(end - start)
            wallclock_time["replay_l2_long"][num_collected].append(end - start)
            obs = similar_transition_replay.observation
            replay_l2_one_seq.append(obs)
            # Replay L2 long-term
            replay_l2_long_seq.append(obs)
            # Replay KL 1-step
            start = time.time()
            similar_transition, similar_indices_kl, kl_div = replay_world_model.sample_similar_distribution(mu.cpu(), logvar.cpu(), action=test_traj["actions"][tr_idx])
            obs = torch.distributions.Normal(similar_transition.mean, similar_transition.sigma).rsample()
            end = time.time()
            replay_kl_one_seq.append(obs)
            replay_kl_div_one.append(kl_div.item())
            # Replay KL long-term
            replay_kl_long_seq.append(obs)
            replay_kl_div_long.append(kl_div.item())
            wallclock_time["replay_kl_one"][num_collected].append(end - start)
            wallclock_time["replay_kl_long"][num_collected].append(end - start)

            for i in range(1, horizon):
                # Base
                real_seq.append(test_traj["observations"][tr_idx + i] / 255.)
                latent, mu, logvar = vae.get_latent((test_traj["observations"][tr_idx + i] / 255.).float().unsqueeze(0).permute(0, 3, 1, 2).to(device))
                real_mu_seq.append(mu)
                real_std_seq.append(torch.sqrt(torch.exp(logvar)))
                # One-step search-based
                start = time.time()
                similar_transition, _ = rollout_world_model.sample_similar(latent.cpu(), return_indices=False, action=test_traj["actions"][tr_idx + i])
                wallclock_time["rollout_one"][num_collected].append(time.time() - start)
                rollout_one_seq.append(similar_transition.next_observation)
                # Long-term search-based
                imagined_obs = []
                # Search best sample for each trajectory
                start = time.time()
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
                wallclock_time["rollout_long"][num_collected].append(time.time() - start)
                rollout_long_seq.append(torch.cat(imagined_obs))
                # One-step SSM
                ssm_context = test_traj["observations"][tr_idx - ssm_context_length + i:tr_idx + i]
                ssm_obs_buffer = [x.permute(2, 0, 1).unsqueeze(0) for x in ssm_context]
                ssm_context, _, _ = vae.get_latent((ssm_context / 255.).float().permute(0, 3, 1, 2).to(device))
                ssm_context_action = test_traj["actions"][tr_idx - 1 + i]
                start = time.time()
                h_t = ssm_short.get_previous_hidden(ssm_context.flatten().unsqueeze(0), ssm_context_action.view(1, -1).to(device), device=device)
                # Step one ahead
                ssm_obs_buffer.pop(0)
                ssm_obs_buffer.append(test_traj["observations"][tr_idx + i].permute(2, 0, 1).unsqueeze(0))
                o_t = torch.cat(ssm_obs_buffer, dim=0)
                o_t, _, _ = vae.get_latent((o_t / 255.).to(device))
                ssm_short_pred, stats_short = ssm_short.predict(h_t, o_t.unsqueeze(0), test_traj["actions"][tr_idx + i].view(1, 1, -1).to(device), device)
                wallclock_time["ssm_one"][num_collected].append(time.time() - start)
                ssm_short_term_seq.append(ssm_short_pred[0])
                ssm_short_stats_seq.append(stats_short)
                # Replay L2 1-step
                start = time.time()
                similar_transition, _ = replay_world_model.sample_similar(latent.cpu(), action=test_traj["actions"][tr_idx + i], k=10)
                wallclock_time["replay_l2_one"][num_collected].append(time.time() - start)
                obs = similar_transition.next_observation
                replay_l2_one_seq.append(obs)
                # Replay L2 long-term
                imagined_obs = []
                start = time.time()
                for tr in similar_indices_replay.squeeze().numpy():
                    imagined_next = replay_world_model.get_transition(torch.tensor([tr + i]))
                    imagined_obs.append(imagined_next.observation)
                wallclock_time["replay_l2_long"][num_collected].append(time.time() - start)
                replay_l2_long_seq.append(torch.cat(imagined_obs))
                # Replay KL 1-step
                start = time.time()
                _, next_mu, next_logvar = vae.get_latent(torch.tensor(test_traj["next_observations"][tr_idx + i] / 255.).float().unsqueeze(0).permute(0, 3, 1, 2).to(device))
                similar_transition, _, kl_div = replay_world_model.sample_similar_distribution(next_mu.cpu(), next_logvar.cpu(), action=test_traj["actions"][tr_idx + i])
                obs = torch.distributions.Normal(similar_transition.mean, similar_transition.sigma).rsample()
                wallclock_time["replay_kl_one"][num_collected].append(time.time() - start)
                replay_kl_one_seq.append(obs)
                replay_kl_div_one.append(kl_div.item())
                # Replay KL long-term
                ref_distrib = torch.distributions.Normal(mu.cpu(), torch.sqrt(torch.exp(logvar)).cpu())
                start = time.time()
                imagined_next = replay_world_model.get_transition(torch.tensor([similar_indices_kl + i]))
                next_distrib = torch.distributions.Normal(imagined_next.mean, imagined_next.sigma)
                obs_next = next_distrib.rsample()
                wallclock_time["replay_kl_long"][num_collected].append(time.time() - start)
                replay_kl_long_seq.append(obs_next.squeeze())
                replay_kl_div_long.append(torch.distributions.kl.kl_divergence(ref_distrib, next_distrib).sum(-1).item())

            # Adjust ssm sequence
            ssm_short_term_seq.pop(-1)
            ssm_short_stats_seq.pop(-1)

            # SSM long-term
            ssm_context = test_traj["observations"][tr_idx - ssm_context_length:tr_idx]
            ssm_obs_buffer = [x.permute(2, 0, 1).unsqueeze(0) for x in ssm_context]
            ssm_context, _, _ = vae.get_latent((ssm_context / 255.).float().permute(0, 3, 1, 2).to(device))
            ssm_context_action = test_traj["actions"][tr_idx - 1]
            start = time.time()
            h_t = ssm_long.get_previous_hidden(ssm_context.flatten().unsqueeze(0), ssm_context_action.view(1, -1).to(device), device=device)
            # Step one ahead
            ssm_obs_buffer.pop(0)
            ssm_obs_buffer.append(test_traj["observations"][tr_idx].permute(2, 0, 1).unsqueeze(0))
            o_t = torch.cat(ssm_obs_buffer, dim=0)
            o_t, _, _ = vae.get_latent((o_t / 255.).to(device))
            ssm_long_term_seq, stats_long = ssm_long.predict(h_t, o_t.unsqueeze(0), test_traj["actions"][tr_idx:tr_idx + horizon].view(1, horizon, -1).to(device), device,
                                                             horizon=horizon)
            end = time.time()
            for _ in range(horizon):
                wallclock_time["ssm_long"][num_collected].append((end - start))
            context_obs, context_stats = ssm_long.get_context_prior(h_t, o_t.unsqueeze(0), test_traj["actions"][tr_idx].view(1, -1).to(device), device)
            ssm_long_term_seq.pop(-1)
            ssm_long_term_seq.insert(0, context_obs)
            stats_long["posterior_mean"].pop(-1)
            stats_long["posterior_logvar"].pop(-1)
            stats_long["posterior_mean"].insert(0, context_stats["posterior_mean"].squeeze(0))
            stats_long["posterior_logvar"].insert(0, context_stats["posterior_logvar"].squeeze(0))

            # Base distribution
            dist_base = torch.distributions.normal.Normal(torch.cat(real_mu_seq), torch.cat(real_std_seq))
            base_seq_tensor = torch.cat([x.unsqueeze(0).permute(0, 3, 1, 2) for x in real_seq]).to(device)
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

            os.makedirs(f"experiments/trajectories_ablation/minerl/{track_name}/samples/{num_collected}/one_step", exist_ok=True)
            os.makedirs(f"experiments/trajectories_ablation/minerl/{track_name}/samples/{num_collected}/long_term", exist_ok=True)
            if save_samples:
                np.savez_compressed(
                    f"experiments/trajectories_ablation/minerl/{track_name}/samples/{num_collected}/one_step/decoded_one_step_sample_{s}.npz",
                    real=np.array([cv2.resize(x.cpu().numpy(), (128, 128)) for x in real_seq]),
                    ssm=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_ssm_short]),
                    rollout=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_rollout_one]),
                    replay_l2=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_l2_one]),
                    replay_kl=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_kl_one])
                )

                np.savez_compressed(
                    f"experiments/trajectories_ablation/minerl/{track_name}/samples/{num_collected}/long_term/decoded_long_term_sample_{s}.npz",
                    real=np.array([cv2.resize(x.cpu().numpy(), (128, 128)) for x in real_seq]),
                    ssm=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_ssm_long]),
                    rollout=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_rollout_long]),
                    replay_l2=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_l2_long]),
                    replay_kl=np.array([cv2.resize(x.cpu().numpy().transpose(1, 2, 0), (128, 128)) for x in decoded_replay_kl_long])
                )

        # Plot heatmaps
        names = ["ssm", "rollout", "replay_l2", "replay_kl"]
        colors = ["tab:red", "tab:blue", "tab:orange", "tab:brown"]
        handles = []
        for measure in ["kl", "l1", "ssim"]:
            # Rollout one-step
            for heatmap_idx, k in enumerate(names):
                mean_kl_dist = torch.cat(kl_divergence_with_base[f"{k}_one"], dim=0).mean(dim=0).cpu().numpy().mean()
                mean_l1_dist = torch.cat(l1_distances_with_base[f"{k}_one"], dim=0).mean(dim=0).cpu().numpy().mean()
                mean_ssim_dist = torch.cat(ssim_with_base[f"{k}_one"], dim=0).mean(dim=0).cpu().numpy().mean()

                heatmap_kl_one_step[heatmap_idx, traj_idx] = mean_kl_dist
                heatmap_l1_one_step[heatmap_idx, traj_idx] = mean_l1_dist
                heatmap_ssim_one_step[heatmap_idx, traj_idx] = mean_ssim_dist
                # Save overall results
                overall_results_kl[f"{k}_one"]["mean"] += mean_kl_dist
                overall_results_l1[f"{k}_one"]["mean"] += mean_l1_dist
                overall_results_ssim[f"{k}_one"]["mean"] += mean_ssim_dist

                # Long-term
                mean_kl_dist = torch.cat(kl_divergence_with_base[f"{k}_long"], dim=0).mean(dim=0).cpu().numpy().mean()
                mean_l1_dist = torch.cat(l1_distances_with_base[f"{k}_long"], dim=0).mean(dim=0).cpu().numpy().mean()
                mean_ssim_dist = torch.cat(ssim_with_base[f"{k}_long"], dim=0).mean(dim=0).cpu().numpy().mean()

                heatmap_kl_long_term[heatmap_idx, traj_idx] = mean_kl_dist
                heatmap_l1_long_term[heatmap_idx, traj_idx] = mean_l1_dist
                heatmap_ssim_long_term[heatmap_idx, traj_idx] = mean_ssim_dist
                # Save overall results
                overall_results_kl[f"{k}_long"]["mean"] += mean_kl_dist
                overall_results_l1[f"{k}_long"]["mean"] += mean_l1_dist
                overall_results_ssim[f"{k}_long"]["mean"] += mean_ssim_dist

            # Wallclock times
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
            short_keys = ["ssm_one", "rollout_one", "replay_l2_one", "replay_kl_one"]
            labels = ["SSM", "Rollout", "Replay-L2", "Replay-KL"]
            colors = ["tab:blue", "tab:red", "tab:orange", "tab:brown"]
            handles = []
            for k, c in zip(short_keys, colors):
                mean_wallclock = np.mean(np.array(wallclock_time[k][num_collected]).reshape(num_samples, horizon), axis=0)
                std_wallclock = np.std(np.array(wallclock_time[k][num_collected]).reshape(num_samples, horizon), axis=0)
                a, = ax[0].plot(range(horizon), mean_wallclock, c=c)
                handles.append(a)
                ax[0].fill_between(range(horizon), mean_wallclock - std_wallclock, mean_wallclock + std_wallclock, alpha=0.2, color=c)

            long_keys = ["ssm_long", "rollout_long", "replay_l2_long", "replay_kl_long"]
            for k, c in zip(long_keys, colors):
                mean_wallclock = np.mean(np.array(wallclock_time[k][num_collected]).reshape(num_samples, horizon), axis=0)
                std_wallclock = np.std(np.array(wallclock_time[k][num_collected]).reshape(num_samples, horizon), axis=0)
                ax[1].plot(range(horizon), mean_wallclock, c=c)
                ax[1].fill_between(range(horizon), mean_wallclock - std_wallclock, mean_wallclock + std_wallclock, alpha=0.2, color=c)

            ax[0].legend(handles=handles, labels=labels, loc="upper left")
            ax[0].set_xticks([0, 5, 10, 15, 20])
            ax[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax[0].set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

            ax[1].set_xticks([0, 5, 10, 15, 20])
            ax[1].set_xticklabels([0, 5, 10, 15, 20])
            ax[1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax[1].set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

            ax[1].set_xlabel("Timestep")
            ax[0].set_ylabel("Time (s)")
            ax[1].set_ylabel("Time (s)")

            fig.savefig(f"experiments/trajectories_ablation/minerl/{track_name}/wallclock_time_{num_collected}.png")
            fig.savefig(f"experiments/trajectories_ablation/minerl/{track_name}/wallclock_time_{num_collected}.pdf")
            fig.clear()

    # Wallclock times
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
    short_keys = ["ssm_one", "rollout_one", "replay_l2_one", "replay_kl_one"]
    labels = ["SSM", "Rollout", "Replay-L2", "Replay-KL"]
    colors = ["tab:blue", "tab:red", "tab:orange", "tab:brown"]
    handles = []
    for k, c in zip(short_keys, colors):
        mean_wallclock = []
        std_wallclock = []
        for v in wallclock_time[k].values():
            mean_wallclock.append(np.mean(v))
            std_wallclock.append(np.std(v))
        mean_wallclock = np.array(mean_wallclock)
        std_wallclock = np.array(std_wallclock)
        a, = ax[0].plot(num_trajectories, mean_wallclock, c=c)
        handles.append(a)
        ax[0].fill_between(num_trajectories, mean_wallclock - std_wallclock, mean_wallclock + std_wallclock, alpha=0.2, color=c)


    long_keys = ["ssm_long", "rollout_long", "replay_l2_long", "replay_kl_long"]
    for k, c in zip(long_keys, colors):
        mean_wallclock = []
        std_wallclock = []
        for v in wallclock_time[k].values():
            mean_wallclock.append(np.mean(v))
            std_wallclock.append(np.std(v))
        mean_wallclock = np.array(mean_wallclock)
        std_wallclock = np.array(std_wallclock)
        ax[1].plot(num_trajectories, mean_wallclock, c=c)
        ax[1].fill_between(num_trajectories, mean_wallclock - std_wallclock, mean_wallclock + std_wallclock, alpha=0.2, color=c)

    ax[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax[0].set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

    ax[1].set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
    ax[1].set_xticklabels(np.array([10, 20, 30, 40, 50, 60, 70, 80]) * max_episode_len)
    ax[1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax[1].set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

    ax[1].set_xlabel("Number of encoded transitions")
    ax[0].set_ylabel("Time (s)")
    ax[1].set_ylabel("Time (s)")

    ax[0].legend(handles=handles, labels=labels, loc="upper left")
    fig.savefig("experiments/trajectories_ablation/minerl/wallclock_time.png")
    fig.savefig("experiments/trajectories_ablation/minerl/wallclock_time.pdf")
    fig.clear()

    # Plot heatmaps
    labels = ["SSM", "Rollout", "Replay-L2", "Replay-KL"]
    maps_one_step = [heatmap_kl_one_step, heatmap_l1_one_step, heatmap_ssim_one_step]
    maps_long_term = [heatmap_kl_long_term, heatmap_l1_long_term, heatmap_ssim_long_term]
    names = ["kl", "l1", "ssim"]
    barlabel = ["KL divergence", "L1 distance", "SSIM"]
    colormaps = ["YlGnBu", "YlGn", "YlOrRd"]
    for m_one, m_long, name, lab, cmap in zip(maps_one_step, maps_long_term, names, barlabel, colormaps):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(17, 7), sharex=True)

        # One-step
        heatmap_one = ax[0].imshow(m_one, cmap=cmap)
        color_threshold = heatmap_one.norm(m_one.max()) / 2
        text_colors = ["black", "white"]

        ax[0].set_xticks(range(len(num_trajectories)), labels=num_trajectories, fontsize=16)
        ax[0].set_yticks(range(len(labels)), labels=labels, fontsize=16)
        ax[0].set_ylabel("Model", fontsize=18)

        # Text annotations
        for h_i in range(len(labels)):
            for h_j in range(len(num_trajectories)):
                ax[0].text(h_j, h_i, round(m_one[h_i, h_j], 2), ha="center", va="center",
                           color=text_colors[int(heatmap_one.norm(m_one[h_i, h_j]) > color_threshold)],
                           fontsize=10.5)

        # Colorbar above first subplot
        pos0 = ax[0].get_position()
        cbar_ax = fig.add_axes([pos0.x0, pos0.y1 + 0.05, pos0.width, 0.05])  # [left, bottom, width, height]
        cbar = fig.colorbar(heatmap_one, cax=cbar_ax, orientation='horizontal')
        cbar.ax.set_title(lab, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Long-term
        heatmap_long = ax[1].imshow(m_long, cmap=cmap)
        color_threshold = heatmap_long.norm(m_long.max()) / 2

        ax[1].set_xticks(range(len(num_trajectories)), labels=num_trajectories, fontsize=16)
        ax[1].set_yticks(range(len(labels)), labels=labels, fontsize=16)
        ax[1].set_xlabel("Number of trajectories", fontsize=18)
        ax[1].set_ylabel("Model", fontsize=18)

        # Text annotations
        for h_i in range(len(labels)):
            for h_j in range(len(num_trajectories)):
                ax[1].text(h_j, h_i, round(m_long[h_i, h_j], 2), ha="center", va="center",
                           color=text_colors[int(heatmap_long.norm(m_long[h_i, h_j]) > color_threshold)],
                           fontsize=10.5)

        fig.savefig(f"experiments/trajectories_ablation/minerl/{track_name}/heatmap_{name}.png", bbox_inches='tight')
