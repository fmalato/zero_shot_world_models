import json
import os
import numpy as np
import torch
import skimage.io as io
#import vizdoom as vzd
import skimage.transform as transform
import cv2

from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from action_mapping import ActionMapper


# Chunked dataset code from: https://discuss.pytorch.org/t/an-iterabledataset-implementation-for-chunked-data/124437/3

class RandomPolicyDataset(Dataset):
    def __init__(self, images):
        super().__init__()
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item]


class StoredDataset(Dataset):
    def __init__(self, prefix_path, file_paths, img_shape=(64, 64, 3), observation_key="observations"):
        self.prefix_path = prefix_path
        self.file_paths = file_paths
        self.img_shape = img_shape
        self.grayscale = True if len(img_shape) == 2 or img_shape[-1] == 1 else False
        self.images = []
        full_file_paths = [os.path.join(self.prefix_path, x) for x in self.file_paths if ".csv" not in x]
        for fp in full_file_paths:
            data = np.load(fp, allow_pickle=True)
            if self.grayscale:
                images = 0.299 * data[observation_key][:, :, :, 0] + 0.587 * data[observation_key][:, :, :, 1] + 0.114 * data[observation_key][:, :, :, 2]
            else:
                images = data[observation_key]
            self.images.append(np.array(images))

        self.images = np.concatenate(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        if img.shape != self.img_shape:
            img = transform.resize(img, self.img_shape)

        if np.max(img) > 1:
            img = img / 255.

        return torch.FloatTensor(img).permute(2, 0, 1)


class StoredDatasetFromVideo(Dataset):
    def __init__(self, prefix_path, file_paths, img_shape=(64, 64, 3), max_frames=None):
        self.prefix_path = prefix_path
        self.file_paths = file_paths
        self.img_shape = img_shape
        self.grayscale = True if len(img_shape) == 2 or img_shape[-1] == 1 else False
        self.images = []

        full_file_paths = [os.path.join(self.prefix_path, x, "recording.mp4") for x in self.file_paths]

        for fp in full_file_paths:
            cap = cv2.VideoCapture(fp)
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (max_frames and count >= max_frames):
                    break
                if self.grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.images.append(frame)
                count += 1
            cap.release()

        self.images = np.array(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]

        # Ensure the image has 3 channels if not grayscale
        if self.grayscale:
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        elif img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        if img.shape != self.img_shape:
            img = transform.resize(img, self.img_shape, anti_aliasing=True)

        if np.max(img) > 1:
            img = img / 255.

        img_tensor = torch.FloatTensor(img)
        if not self.grayscale:
            img_tensor = img_tensor.permute(2, 0, 1)
        else:
            img_tensor = img_tensor.permute(2, 0, 1)  # (1, H, W)

        return img_tensor


class StoredDatasetWithHistory(Dataset):
    def __init__(self, prefix_path, file_paths, img_shape=(64, 64), history_length=4):
        self.prefix_path = prefix_path
        self.file_paths = file_paths
        self.img_shape = img_shape
        self.history_length = history_length
        self.images = []
        full_file_paths = [os.path.join(self.prefix_path, x) for x in self.file_paths if ".csv" not in x]
        for fp in full_file_paths:
            data = np.load(fp, allow_pickle=True)
            images = 0.299 * data["observations"][:, :, :, 0] + 0.587 * data["observations"][:, :, :, 1] + 0.114 * data["observations"][:, :, :, 2]
            self.images.append(self._compute_history_obs(images))

        self.images = np.concatenate(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]

        return torch.FloatTensor(img)

    def _compute_history_obs(self, images):
        images_with_history = []
        obs_buffer = [np.zeros(shape=self.img_shape) for _ in range(self.history_length)]
        for i in range(images.shape[0]):
            obs_buffer.append(images[i])
            obs_buffer.pop(0)
            images_with_history.append(np.array(obs_buffer))

        return np.array(images_with_history)


class VizdoomDataset(Dataset):
    def __init__(self, prefix_path, file_paths=None, img_shape=(64, 64, 3), resize=True, grayscale=False):
        import matplotlib.pyplot as plt
        self.prefix_path = prefix_path
        self.file_paths = file_paths if file_paths is not None else os.listdir(prefix_path)
        self.img_shape = img_shape
        self.images = []
        full_file_paths = [os.path.join(self.prefix_path, x) for x in self.file_paths if ".csv" not in x]

        game = vzd.DoomGame()
        game.load_config(f"{prefix_path.split(sep='/')[-1]}.cfg")
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_window_visible(False)
        game.set_render_hud(False)

        game.init()

        for fp in tqdm(full_file_paths, desc="Encoding Vizdoom trajectories..."):
            game.replay_episode(fp)
            obs = []
            while not game.is_episode_finished():
                s = game.get_state().screen_buffer
                game.advance_action()
                if resize:
                    obs.append(np.array(Image.fromarray(s.swapaxes(0, 2)).resize(size=(64, 64))).swapaxes(0, 2).astype(np.float32) / 255.)
                else:
                    obs.append(s.astype(np.float32) / 255.)

            obs = np.array(obs)
            if grayscale:
                obs = 0.299 * obs[:, :, 0] + 0.587 * obs[:, :, 1] + 0.114 * obs[:, :, 2]
            self.images.append(np.array(obs))

        self.images = np.concatenate(self.images)
        game.close()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]

        return torch.FloatTensor(img)


class FullAtariGameDataset(Dataset):
    def __init__(self, prefix_path, game_name, img_shape=(64, 64, 3), train=True):
        self.prefix_path = prefix_path
        self.split = "train" if train else "test"
        self.img_shape = img_shape
        self.grayscale = True if len(img_shape) == 2 or img_shape[-1] == 1 else False
        self.images = []
        self.split = "train" if train else "test"
        self.game_name = game_name

        images_path = os.path.join(os.path.join(self.prefix_path, game_name), self.split)
        self.images = [os.path.join(images_path, x) for x in os.listdir(images_path) if ".json" not in x]

        """#with open(os.path.join(images_path, "actions.json"), "r") as f:
            self.actions = json.load(f)
        f.close()"""

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = io.imread(self.images[item]) / 255.
        if self.grayscale:
            img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        img = transform.resize(img, self.img_shape)
        #frame_id = self.images[item].split(sep=".")[0]
        #action = int(self.actions[frame_id])

        return torch.FloatTensor(img).permute(2, 0, 1)


class StoredDatasetFullSuite(Dataset):
    def __init__(self, prefix_path, suite_name, img_shape=(64, 64, 3), observation_key="observations"):
        self.prefix_path = prefix_path
        self.suite_name = suite_name
        self.img_shape = img_shape
        self.grayscale = True if len(img_shape) == 2 or img_shape[-1] == 1 else False
        self.images = []
        full_file_paths = self._discover_image_paths()
        for fp in full_file_paths:
            data = np.load(fp, allow_pickle=True)
            if self.grayscale:
                data[observation_key] = 0.299 * data[observation_key][:, :, 0] + 0.587 * data[observation_key][:, :, 1] + 0.114 * data[observation_key][:, :, 2]
            self.images.append(np.array(data[observation_key]))

        self.images = np.concatenate(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        if img.shape != self.img_shape:
            img = transform.resize(img, self.img_shape)

        if np.max(img) > 1:
            img = img / 255.

        return torch.FloatTensor(img).permute(2, 0, 1)

    def _discover_image_paths(self):
        suite_path = os.path.join(self.prefix_path, self.suite_name)
        full_game_paths = []
        for game_name in os.listdir(suite_path):
            game_path = os.path.join(suite_path, game_name)
            traj_list = [os.path.join(game_path, x) for x in os.listdir(game_path) if x.endswith(".npz")]
            full_game_paths.append(traj_list)

        return list(np.concatenate(full_game_paths))


class ExpertExperienceWorldModel(Dataset):
    def __init__(self, data_path, file_names, suite="atari"):
        self.data_path = data_path
        self.suite_name = suite
        self.trajectories_names = os.listdir(data_path) if file_names is None else file_names
        self.action_mapper = ActionMapper(game=self.data_path.split(sep="/")[-1]) if suite == "atari" else None

        self.images = []
        self.actions = []
        for tn in self.trajectories_names:
            if not tn.endswith(".npz"):
                tn += ".npz"
            data = np.load(os.path.join(self.data_path, f"{tn}"), allow_pickle=True)
            self.images.append(data["observations"])
            if self.action_mapper:
                actions = [self.action_mapper.map_ale_to_gym(a) for a in data["actions"]]
                self.actions.append(np.array(actions))
            else:
                self.actions.append(data["actions"])

        self.images = np.concatenate(self.images)
        self.actions = np.concatenate(self.actions)

    def __len__(self):
        return len(self.images) - 1

    def __getitem__(self, item):
        return {"obs": torch.FloatTensor(self.images[item]).permute(2, 0, 1), "act": torch.LongTensor([self.actions[item]]), "next_obs": torch.FloatTensor(self.images[item + 1]).permute(2, 0, 1)}

    def _discover_image_paths(self):
        suite_path = os.path.join(self.data_path, self.suite_name)
        full_game_paths = []
        for game_name in os.listdir(suite_path):
            game_path = os.path.join(suite_path, game_name)
            traj_list = [os.path.join(game_path, x) for x in os.listdir(game_path) if x.endswith(".npz")]
            full_game_paths.append(traj_list)

        return list(np.concatenate(full_game_paths))


class ExpertExperienceWorldModelFullSuite(Dataset):
    def __init__(self, data_path, file_names, suite="atari"):
        self.data_path = data_path
        self.trajectories_names = os.listdir(data_path) if file_names is None else file_names
        self.action_mapper = ActionMapper(game=self.data_path.split(sep="/")[-1]) if suite == "atari" else None

        self.images = []
        self.actions = []
        for tn in self.trajectories_names:
            if not tn.endswith(".npz"):
                tn += ".npz"
            data = np.load(os.path.join(self.data_path, f"{tn}"), allow_pickle=True)
            self.images.append(data["observations"])
            if self.action_mapper:
                actions = [self.action_mapper.map_ale_to_gym(a) for a in data["action"]]
                self.actions.append(np.array(actions))
            else:
                self.actions.append(data["actions"])

        self.images = np.concatenate(self.images)
        self.actions = np.concatenate(self.actions)

    def __len__(self):
        return len(self.images) - 1

    def __getitem__(self, item):
        return {"obs": torch.FloatTensor(self.images[item]).permute(2, 0, 1), "act": torch.LongTensor([self.actions[item]]), "next_obs": torch.FloatTensor(self.images[item + 1]).permute(2, 0, 1)}


if __name__ == '__main__':
    base_path = "vizdoom/deadly_corridor"
    data = VizdoomDataset(prefix_path=base_path,)
    obs = data.__getitem__(123)
