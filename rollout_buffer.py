from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    observation: torch.Tensor
    action: torch.Tensor
    next_observation: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor


class RolloutBuffer:
    def __init__(
            self,
            num_collected: int,
            obs_size: int,
            num_actions: int,
            max_episode_len: int,
            gamma: float,
            context_length: int = 1,
            is_minerl: bool = False,
    ):
        self.observations = torch.zeros((num_collected, max_episode_len, obs_size), dtype=torch.float32)
        self.actions = torch.zeros((num_collected, max_episode_len, 1 if not is_minerl else 10), dtype=torch.float32)
        self.next_observations = torch.zeros((num_collected, max_episode_len, obs_size), dtype=torch.float32)
        self.means = torch.zeros((num_collected, max_episode_len, obs_size), dtype=torch.float32)
        self.logvars = torch.zeros((num_collected, max_episode_len, obs_size), dtype=torch.float32)
        self.lengths = np.zeros((num_collected, ), dtype=np.int32)

        self.num_collected = num_collected
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.max_episode_len = max_episode_len

        self._gammas = torch.tensor([gamma ** t for t in range(max_episode_len)], dtype=torch.float32)
        self._current_index = 0
        self._context_length = context_length
        self._full = False
        self._is_minerl = is_minerl

    def add_trajectory(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            next_observations: torch.Tensor,
            means: torch.Tensor,
            logvars: torch.Tensor
    ):
        if self._current_index >= self.num_collected:
            self._current_index = 0
            self._full = True

        self.observations[self._current_index] = torch.zeros((self.max_episode_len, self.obs_size), dtype=torch.float32)
        self.actions[self._current_index] = torch.zeros((self.max_episode_len, 1 if not self._is_minerl else 10), dtype=torch.float32)
        self.next_observations[self._current_index] = torch.zeros((self.max_episode_len, self.obs_size), dtype=torch.float32)
        self.means[self._current_index] = torch.zeros((self.max_episode_len, self.obs_size), dtype=torch.float32)
        self.logvars[self._current_index] = torch.zeros((self.max_episode_len, self.obs_size), dtype=torch.float32)
        self.lengths[self._current_index] = 0

        traj_length = int(actions.size(0))
        self.observations[self._current_index, :traj_length] = observations
        self.actions[self._current_index, :traj_length] = actions
        self.next_observations[self._current_index, :traj_length] = next_observations
        self.means[self._current_index, :traj_length] = means
        self.logvars[self._current_index, :traj_length] = logvars
        self.lengths[self._current_index] = traj_length

        self._current_index += 1

    def get_trajectory(self, index):
        data = (
            self.observations[index],
            self.actions[index],
            self.next_observations[index],
            self.means[index],
            self.logvars[index]
        )

        return Transition(*tuple(map(self.to_torch, data)))

    def sample_similar(self, latent, return_indices=False, action=None):
        batch_size = 1
        sampled_obs = torch.zeros(size=(batch_size, self.num_collected, self.obs_size), dtype=torch.float32)
        sampled_actions = torch.zeros(size=(batch_size, self.num_collected, 1 if not self._is_minerl else 10), dtype=torch.float32)
        sampled_next_obs = torch.zeros(size=(batch_size, self.num_collected, self.obs_size), dtype=torch.float32)
        sampled_means = torch.zeros(size=(batch_size, self.num_collected, self.obs_size), dtype=torch.float32)
        sampled_logvars = torch.zeros(size=(batch_size, self.num_collected, self.obs_size), dtype=torch.float32)

        closest_indices = torch.zeros((batch_size, self.num_collected), dtype=torch.long)

        for i in range(batch_size):
            # Sample the best from each trajectory, exactly with the specified action
            for n in range(self.num_collected):
                distances = torch.cdist(self.observations[n, :self.lengths[n]], latent[i].unsqueeze(0), p=2) ** 2
                closest_index = torch.topk(-distances, k=1, dim=0)[1].T
                closest_indices[i, n] = closest_index

                sampled_obs[i, n] = self.observations[n, closest_index]
                sampled_actions[i, n] = self.actions[n, closest_index]
                sampled_next_obs[i, n] = self.next_observations[n, closest_index]
                sampled_means[i, n] = self.means[n, closest_index]
                sampled_logvars[i, n] = self.logvars[n, closest_index]

        if action is not None:
            if not self._is_minerl:
                ok_indices = torch.argwhere(sampled_actions.squeeze() == action).squeeze(-1)
            else:
                ok_indices = torch.argwhere((sampled_actions.squeeze() == action).float().sum(-1) == 0)
            # If there's not enough points to estimate, estimate the regular state instead
            if ok_indices.size(0) > self.num_collected / 2:
                sampled_obs = sampled_obs[:, ok_indices, :]
                sampled_actions = sampled_actions[:, ok_indices, :]
                sampled_next_obs = sampled_next_obs[:, ok_indices, :]
                sampled_means = sampled_means[:, ok_indices, :]
                sampled_logvars = sampled_logvars[:, ok_indices, :]

        data = (
            sampled_obs,
            sampled_actions,
            sampled_next_obs,
            sampled_means,
            sampled_logvars,
        )

        return Transition(*tuple(map(self.to_torch, data))), closest_indices if return_indices else None


    def get_transition(self, traj_index: torch.Tensor, transition_index: torch.Tensor):
        # Ensure sampled index is bounded within trajectory length limits.
        batched_lengths = torch.from_numpy(self.lengths[traj_index.numpy()] - 1)
        if transition_index.ndim > 1:
            batched_lengths = batched_lengths.unsqueeze(-1).repeat(1, transition_index.size(-1))
            traj_index = traj_index.unsqueeze(-1)
        transition_index = torch.min(transition_index, batched_lengths).long()
        transition_index = torch.max(transition_index, torch.zeros_like(transition_index)).long()

        data = (
            self.observations[traj_index, transition_index],
            self.actions[traj_index, transition_index],
            self.next_observations[traj_index, transition_index],
            self.means[traj_index, transition_index],
            self.logvars[traj_index, transition_index]
        )

        return Transition(*tuple(map(self.to_torch, data)))

    def to_torch(self, data):
        return torch.Tensor(data)

    def plan(self, latent: torch.Tensor, horizon: int = 20, deterministic: bool = True):
        action_probs = []

        closest_indices = torch.zeros((self.num_collected,), dtype=torch.long)

        # Sample the best from each trajectory, exactly with the specified action
        for n in range(self.num_collected):
            distances = torch.cdist(self.observations[n, :self.lengths[n]], latent.unsqueeze(0), p=2) ** 2
            closest_index = torch.topk(-distances.flatten(), k=1, dim=0)[1]
            closest_indices[n] = closest_index

        for h in range(horizon):
            sampled_actions = [int(self.actions[i, (closest_indices[i] + h) % self.lengths[i]].item()) for i in range(self.num_collected)]
            action_probs.append(torch.bincount(torch.Tensor(sampled_actions).long(), minlength=self.num_actions) / self.num_collected)

        action_probs = torch.stack(action_probs)
        if deterministic:
            return list(torch.argmax(action_probs, dim=1).numpy())
        else:
            act_distribution = torch.distributions.Categorical(probs=action_probs)
            return act_distribution.sample()