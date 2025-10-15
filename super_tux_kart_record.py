import os

import pystk
import pygame
import numpy as np
import cv2
import matplotlib.pyplot as plt

from super_tux_env import SuperTuxEnv

ACTIONS = {
    "0,0,0": 0,
    "0,0,1": 1,
    "0,0,-1": 2,
    "0,1,0": 3,
    "0,1,1": 4,
    "0,1,-1": 5,
    "0.5,0,0": 6,
    "0.5,0,1": 7,
    "0.5,0,-1": 8
}


if __name__ == '__main__':
    NUM_RACES = 35
    STARTING_OFFSET = 15
    NUM_LAPS = 1
    #TRACKS = ["lighthouse", "fortmagma", "sandtrack", "snowmountain", "volcano", "zengarden"]
    TRACKS = ["fortmagma"]
    RENDER = True
    reward_scheme = "nodes"
    use_time_reward = True

    for t in TRACKS:
        pygame.init()
        env = SuperTuxEnv(track_name=t, num_laps=NUM_LAPS, render=RENDER, use_time_reward=use_time_reward,
                          reward_scheme=reward_scheme, screen_width=800, screen_height=600, obs_size=(800, 600))

        for i in range(STARTING_OFFSET, STARTING_OFFSET + NUM_RACES):
            obs, _ = env.reset()

            acceleration_state = 0
            brake_state = 0
            steer_state = 0

            frames = []
            actions = []
            rewards = []
            cumulative_reward = 0.0
            race_done = False
            first_step = False
            braking = False
            brake_counter = 5

            while not race_done:
                key_events = [x for x in pygame.event.get() if x.type == pygame.KEYDOWN or x.type == pygame.KEYUP]
                for event in key_events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_w:
                            acceleration_state = 0.5
                        if event.key == pygame.K_s:
                            acceleration_state = 0
                            brake_state = 1
                        if event.key == pygame.K_d:
                            steer_state = 1
                        if event.key == pygame.K_a:
                            steer_state = -1
                        elif event.key == pygame.K_q:
                            race_done = True
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_w:
                            acceleration_state = 0
                        if event.key == pygame.K_s:
                            brake_state = 0
                        if event.key == pygame.K_d:
                            steer_state = 0
                        if event.key == pygame.K_a:
                            steer_state = 0

                action = pystk.Action(acceleration=acceleration_state, brake=brake_state, steer=steer_state)
                actions.append(ACTIONS[f"{acceleration_state},{brake_state},{steer_state}"])

                obs, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += reward
                frames.append(cv2.resize(obs, (64, 64)))
                rewards.append(reward)

                if RENDER:
                    env.render(mode="human")

                race_done = terminated or truncated

            #os.makedirs(f"super_tux_kart/trajectories_vae/{t}", exist_ok=True)
            #np.savez_compressed(f"super_tux_kart/trajectories_vae/{t}/race_{i}.npz", observations=frames, actions=np.array(actions), rewards=np.array(rewards))
