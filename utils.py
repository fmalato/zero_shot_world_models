import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt


def freeze_parameters(net):
    for _, params in net.named_parameters():
        params.requires_grad = False


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def save_video(frames, path, name):
    """
    Saves a video containing frames.
    """
    # Preprocessing
    if np.max(frames) <= 1:
        frames = (frames * 255).clip(0, 255).astype('uint8')
    if frames.shape[-1] != 3:
        frames = frames.transpose(0, 3, 1, 2)

    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(
        str(pathlib.Path(path)/f'{name}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), 25., (W, H), True
    )
    for frame in frames[..., ::-1]:
        writer.write(frame)
    writer.release()


def plot_images(data_path: str, labels: list, save_path: str):
    # Parameters
    n_rows = 5
    n_cols = 20
    img_w, img_h = 256, 256
    pad_vertical = 50

    # Margins
    left_margin = 120
    bottom_margin = 160
    right_margin = 50
    top_margin = 40

    data = np.load(data_path)
    sequences = [data[k] for k in data.files]
    sequences_new = []
    # Re-adjust indices
    for s in sequences[1:]:
        s = list(s)
        s.insert(0, data["real"][0])
        s.pop(-1)
        sequences_new.append(np.array(s))

    sequences_new.insert(0, sequences[0])
    sequences = sequences_new

    # Compute figure size in pixels
    fig_width_px = left_margin + n_cols * img_w + right_margin
    fig_height_px = bottom_margin + n_rows * img_h + pad_vertical * (n_rows - 1) + top_margin

    # Create figure
    fig = plt.figure(figsize=(fig_width_px / 100, fig_height_px / 100))

    # Plot images
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            x0 = left_margin + col_idx * img_w
            y0 = bottom_margin + (n_rows - 1 - row_idx) * (img_h + pad_vertical)

            # Convert to relative coordinates
            left = x0 / fig_width_px
            bottom = y0 / fig_height_px
            width = img_w / fig_width_px
            height = img_h / fig_height_px

            ax = fig.add_axes([left, bottom, width, height])
            ax.imshow(sequences[row_idx][col_idx])
            ax.axis('off')

    for i, label in enumerate(labels):
        y_center_px = bottom_margin + (n_rows - 1 - i) * (img_h + pad_vertical) + img_h / 2
        fig.text((left_margin - 20) / fig_width_px, y_center_px / fig_height_px,
                 label, va='center', ha='right', rotation=90, fontsize=42)

    # Add x-axis label ("Time")
    fig.text(0.5, (bottom_margin - 140) / fig_height_px, "Timestep", ha='center', fontsize=56)

    # Add ticks every 5th image with large font
    for i in range(n_cols):
        if i % 5 == 0:
            x_center_px = left_margin + i * img_w + img_w / 2
            fig.text(x_center_px / fig_width_px, (bottom_margin - 70) / fig_height_px,
                     str(i), ha='center', fontsize=60)

    for row_idx in [0, 1]:  # Between row 0-1 and 1-2
        y_px = bottom_margin + (n_rows - row_idx) * (img_h + pad_vertical) - pad_vertical / 2
        y_rel = y_px / fig_height_px
        line = Line2D(
            [left_margin / fig_width_px, (left_margin + n_cols * img_w) / fig_width_px],
            [y_rel, y_rel],
            color='black',
            linewidth=3
        )
        fig.add_artist(line)

    fig.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    import os
    from matplotlib.lines import Line2D

    experiment = "trajectories_ablation"
    env = "super_tux_kart"
    tracks = [x for x in os.listdir(f"experiments/{experiment}/{env}/") if ".png" not in x]
    labels = ["Real", "SSM", "Rollout", "L2", "KL"]

    for t in tracks:
        print(f"Working on {t}")
        for l in [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]:
            for samples_dir in ["one_step", "long_term"]:
                samples_path = f"experiments/{experiment}/{env}/{t}/{l}/samples/{samples_dir}"
                os.makedirs(f"{samples_path}/images/{samples_dir}", exist_ok=True)
                for i, el in enumerate([x for x in os.listdir(samples_path) if x.endswith(".npz")]):
                    print(f"[{t}] Working on {el}")
                    plot_images(f"{samples_path}/{el}", labels, save_path=f"{samples_path}/images/{samples_dir}/sample_{i}.png")
