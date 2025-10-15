import numpy as np
import pygame
import pystk

from super_tux_env import SuperTuxEnv

# Optional
import torch
from world_model import VAE, freeze_parameters
from unet_representation import UNet


if __name__ == '__main__':
    track_name = "lighthouse"
    render_data = "image"
    encoder_name = f"unet_50_256"
    record_video = True

    env = SuperTuxEnv(track_name, render=True, render_data=render_data, num_laps=1)
    pygame.init()

    race_done = False
    close_pressed = False
    needs_rescue = False
    total_reward = 0.0

    acceleration_state = 0
    brake_state = 0
    steer_state = 0

    if record_video:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #encoder = VAE(input_shape=(1, 3, 64, 64), latent_size=256)
        #encoder.load_state_dict(torch.load(f"vae_models/super_tux_kart/{track_name}/{encoder_name}.pth"))
        encoder = UNet(in_channels=3, out_channels=3, bottleneck_size=256)
        encoder.load_state_dict(torch.load(f"unet_models/{track_name}/{encoder_name}.pth"))
        encoder.to(device)
        freeze_parameters(encoder)
        encoder.eval()
        frames = []
        residuals = []
        images = []

    obs, _ = env.reset()

    while not race_done:
        key_events = [x for x in pygame.event.get() if x.type == pygame.KEYDOWN or x.type == pygame.KEYUP]
        for event in key_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    acceleration_state = 1
                if event.key == pygame.K_s:
                    brake_state = 0.05
                if event.key == pygame.K_d:
                    steer_state = 1
                if event.key == pygame.K_a:
                    steer_state = -1
                elif event.key == pygame.K_q:
                    close_pressed = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    acceleration_state = 0
                if event.key == pygame.K_s:
                    brake_state = 0
                if event.key == pygame.K_d:
                    steer_state = 0
                if event.key == pygame.K_a:
                    steer_state = 0

        action = pystk.Action(acceleration=acceleration_state, brake=brake_state, steer=steer_state, rescue=needs_rescue)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if record_video:
            images.append(obs)
            #frames.append(encoder.get_latent(torch.tensor(obs / 255., dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device))[0])
            frame, previous = encoder.encode(torch.tensor(obs / 255., dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device))
            frames.append(frame)
            residuals.append(previous)

        env.render(mode="human")
        race_done = terminated or truncated or close_pressed
        needs_rescue = info["needs_rescue"]

    print(f"Total reward: {total_reward}")
    env.close()
    pygame.quit()

    if record_video:
        import cv2

        # Define the output video file name, codec, frame rate, and frame size
        output_file = f'output_video_compared_{track_name}_{encoder_name}_all_residuals.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use different codecs like 'MJPG', 'XVID', etc.
        fps = 30  # Frame rate (frames per second)

        # Create a VideoWriter object
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (7 * 800, 600))

        # Write each frame to the video file
        for real, frame, res in zip(images, frames, residuals):
            img = encoder.decode_all_layers(frame, res)

            real_frame = cv2.resize(real, (800, 600))
            real_frame = cv2.cvtColor(real_frame, cv2.COLOR_BGR2RGB)
            partial = real_frame
            for i in img:
                img = cv2.resize(i.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255., (800, 600))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                partial = np.hstack((partial, img))
            video_writer.write(partial.astype(np.uint8))

        # Release the video writer once done
        video_writer.release()