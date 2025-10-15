import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pygame
import pystk

from super_tux_env import SuperTuxEnv

# Optional
import torch
import torch.nn.functional as F
import scipy

from world_model import VAE
from replay_buffer import ReplayBuffer
from utils import freeze_parameters


ACTIONS = {
    0: [0, 0, 0],
    1: [0, 0, 1],
    2: [0, 0, -1],
    3: [0, 0.05, 0],
    4: [0, 0.05, 1],
    5: [0, 0.05, -1],
    6: [1, 0, 0],
    7: [1, 0, 1],
    8: [1, 0, -1]
}


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


if __name__ == '__main__':
    track_name = "snes_rainbowroad"
    render_data = "image"
    encoder_name = f"encoder_128_250_{render_data}"
    record_video = False
    render = False
    use_time_reward = False
    num_laps = 1
    num_races = 50
    selectable_delay = 0

    pygame.init()
    env = SuperTuxEnv(track_name, render=render, render_data=render_data, num_laps=num_laps, use_time_reward=use_time_reward)

    race_done = False
    close_pressed = False
    needs_rescue = False
    total_reward = 0.0

    acceleration_state = 0
    brake_state = 0
    steer_state = 0

    k = 10
    num_collected = 10
    max_episode_len = 2500
    latent_size = 128
    num_actions = 9
    planning_horizon = 20

    traj_path = f"super_tux_kart/trajectories/{track_name}/train"

    replay_world_model = ReplayBuffer(
        capacity=num_collected * max_episode_len,
        obs_size=latent_size,
        num_actions=num_actions
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VAE(input_shape=(1, 3, 64, 64), latent_size=latent_size)
    encoder.load_state_dict(torch.load(f"vae_models/super_tux_kart/{track_name}/{encoder_name}.pth"))
    encoder.to(device)
    freeze_parameters(encoder)
    encoder.eval()

    latents = []
    actions = []

    traj_path = f"super_tux_kart/trajectories/{track_name}/train"
    for traj in os.listdir(traj_path)[:num_collected]:
        data = np.load(f"{traj_path}/{traj}", allow_pickle=True)
        latents, means, logvars = encoder.get_latent(torch.tensor(data["observations"] / 255.).float().permute(0, 3, 1, 2).to(device))
        actions = torch.from_numpy(data["actions"]).float().unsqueeze(-1)
        latents_next = torch.cat([latents[1:], torch.zeros(size=(1, latent_size)).to(device)], dim=0)
        for l, a, l_next, m, lg in zip(latents, actions, latents_next, means, logvars):
            replay_world_model.add_transition(l, a, l_next, m, lg)

    best_reward = 0.0
    reward_history = []
    avg_rewards_history = np.zeros(shape=(num_races,))

    os.makedirs(f"experiments/planning/{track_name}", exist_ok=True)
    #with open(f"experiments/planning/{track_name}/replay_KL.csv", "a+") as f:
    #    f.write("k,mean,std,ci_95_low,ci_95_high,ci_99_low,ci_99_high\n")

    reward_history = []
    delta_ts = []
    for r in range(num_races):
        obs, _ = env.reset()
        timestep = 0
        current_indices = []
        race_done = False
        total_reward = 0.0
        planned_actions = []

        while not race_done:
            reference_latent, mean, logvar = encoder.get_latent(
                torch.tensor(obs / 255., dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device))
            if len(planned_actions) == 0:
                start = time.time()
                planned_actions = replay_world_model.plan_distributional(mean.cpu(), logvar.cpu(), horizon=planning_horizon)
                delta_t = time.time() - start
                delta_ts.append(delta_t)

            current_action = planned_actions.pop(0)
            current_action = ACTIONS[current_action]
            acceleration_state = current_action[0]
            brake_state = current_action[1]
            steer_state = current_action[2]

            action = pystk.Action(acceleration=acceleration_state, brake=brake_state, steer=steer_state,
                                  rescue=needs_rescue)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if render:
                env.render(mode="human")
            race_done = terminated or truncated or close_pressed
            needs_rescue = info["needs_rescue"]

            timestep += 1

        reward_history.append(total_reward)
        avg_reward = float(np.mean(reward_history))
        avg_rewards_history[r] = avg_reward

        if r % 10 == 0:
            print(f"[Iteration {r} - k = {k}] Avg. reward: {float(np.mean(reward_history)):.4f} | Last reward: {reward_history[-1]:.4f}")

    mean, lower_95, upper_95 = mean_confidence_interval(reward_history, confidence=0.95)
    _, lower_99, upper_99 = mean_confidence_interval(reward_history, confidence=0.99)

    print(f"[k = {k}] Final performance: {float(np.mean(reward_history)):.4f} +/- {float(np.std(reward_history)):.4f} CI_95: [{lower_95:.4f}, {upper_95:.4f}] | CI_99: [{lower_99:.4f}, {upper_99:.4f}]")
    #with open(f"experiments/planning/{track_name}/replay_KL.csv", "a+") as f:
    #    f.write(f"{k},{float(np.mean(reward_history)):.4f},{float(np.std(reward_history)):.4f},{lower_95:.4f},{upper_95:.4f},{lower_99:.4f},{upper_99:.4f}\n")
    print(f"Track: {track_name} - Inference time: {np.mean(delta_ts)} +/- {np.std(delta_ts)}")

    env.close()
    pygame.quit()