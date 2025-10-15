import os

import cv2
import numpy as np


def compute_similarity(frames_1, frames_2):
    l1 = frames_1.shape[0]
    l2 = frames_2.shape[0]

    if l1 > l2:
        additional_frames = np.zeros(shape=(l1 - l2, *frames_2.shape[1:]))
        frames_2 = np.concatenate([frames_2, additional_frames], axis=0)
    elif l1 < l2:
        additional_frames = np.zeros(shape=(l2 - l1, *frames_1.shape[1:]))
        frames_1 = np.concatenate([frames_1, additional_frames], axis=0)

    residual_frames = np.abs(frames_1 - frames_2)
    score = np.mean(np.mean(np.mean(residual_frames, axis=-1), axis=-1), axis=-1)

    return np.mean(score)


def video_to_frames(fp):
    cap = cv2.VideoCapture(os.path.join(fp, "recording.mp4"))
    count = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (None and count >= None):
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    return np.array(frames)


def frames_to_video(frames):
    w, h = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.avi', fourcc, 1, (w, h))

    for j in range(frames.shape[0]):
        video.write(frames[j])

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    #track_name = 'volcano_island'
    #data_path = f'super_tux_kart/trajectories/{track_name}/train'
    data_path = 'minerl/MineRLTreechop-v0'

    pixel_residual = True

    num_trajectories = 10
    trajectories = os.listdir(data_path)[:num_trajectories]

    scores = []
    for i, t1 in enumerate(trajectories):
        for j, t2 in enumerate(trajectories):
            if i != j:
                print(f"Working on pair {i}, {j}")
                if pixel_residual:
                    if 'minerl' in data_path:
                        video_1 = video_to_frames(os.path.join(data_path, t1))
                        video_2 = video_to_frames(os.path.join(data_path, t2))
                    else:
                        video_1 = np.load(os.path.join(data_path, t1), allow_pickle=True)["observations"]
                        video_2 = np.load(os.path.join(data_path, t2), allow_pickle=True)["observations"]
                    scores.append(compute_similarity(video_1, video_2))
                else:
                    if 'super_tux_kart' in data_path:
                        video_1 = np.load(os.path.join(data_path, t1), allow_pickle=True)["observations"]
                        video_2 = np.load(os.path.join(data_path, t2), allow_pickle=True)["observations"]
                        video_1 = frames_to_video(video_1)
                        video_2 = frames_to_video(video_2)



    print(f'Mean score: {np.mean(scores)}')
