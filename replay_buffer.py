import torch

from typing import NamedTuple


class Transition(NamedTuple):
    observation: torch.Tensor
    action: torch.Tensor
    next_observation: torch.Tensor
    mean: torch.Tensor
    sigma: torch.Tensor


class ReplayBuffer:
    def __init__(
            self,
            capacity: int,
            obs_size: int,
            num_actions: int,
            is_minerl: bool = False
    ):
        self.observations = torch.zeros(size=(capacity, obs_size))
        self.actions = torch.zeros(size=(capacity, 1 if not is_minerl else 10))
        self.next_observations = torch.zeros(size=(capacity, obs_size))
        self.means = torch.zeros(size=(capacity, obs_size))
        self.sigmas = torch.zeros(size=(capacity, obs_size))

        self._current_idx = 0
        self._is_minerl = is_minerl
        self.capacity = capacity
        self.num_actions = num_actions

    def add_transition(self, o, a, o_next, m, l):
        self.observations[self._current_idx] = o
        self.actions[self._current_idx] = a
        self.next_observations[self._current_idx] = o_next
        self.means[self._current_idx] = m
        self.sigmas[self._current_idx] = torch.sqrt(torch.exp(l))

        self._current_idx = (self._current_idx + 1) % self.capacity

    def sample_similar(self, latent: torch.Tensor, action: torch.Tensor = None, k: int = 1):
        distances = torch.cdist(self.observations[:self._current_idx], latent, p=2)
        if action is not None:
            if self._is_minerl:
                ok_indices = torch.argwhere((self.actions[:self._current_idx].squeeze() == action).long().sum(-1) == self.actions.size(-1)).squeeze()
            else:
                ok_indices = torch.argwhere(self.actions[:self._current_idx].squeeze() == action)
            if ok_indices.numel() >= k:
                distances = distances[ok_indices]
        indices = torch.topk(-distances, k=k, dim=0).indices.squeeze()

        data = (
            self.observations[indices],
            self.actions[indices],
            self.next_observations[indices],
            self.means[indices],
            self.sigmas[indices]
        )

        return Transition(*tuple(map(self.to_torch, data))), indices

    def sample_similar_distribution(self, mean: torch.Tensor, logvar: torch.Tensor, action: torch.Tensor = None):
        if self._current_idx < self.capacity:
            stored_dists = torch.distributions.Normal(self.means[:self._current_idx], self.sigmas[:self._current_idx] + 1e-8)
            batched_dist = torch.distributions.Normal(mean.repeat(self._current_idx, 1), torch.sqrt(torch.exp(logvar)).repeat(self._current_idx, 1) + 1e-8)
            distances = torch.distributions.kl.kl_divergence(stored_dists, batched_dist).sum(-1)
        else:
            stored_dists = torch.distributions.Normal(self.means, self.sigmas + 1e-8)
            batched_dist = torch.distributions.Normal(mean.repeat(self.means.size(0), 1), torch.sqrt(torch.exp(logvar)).repeat(self.sigmas.size(0), 1) + 1e-8)
            distances = torch.distributions.kl.kl_divergence(stored_dists, batched_dist).sum(-1)

        # Action conditioning
        if action is not None:
            if self._is_minerl:
                ok_indices = torch.argwhere((self.actions[:self._current_idx].squeeze() == action).long().sum(-1) == self.actions.size(-1)).squeeze()
            else:
                ok_indices = torch.argwhere(self.actions[:self._current_idx].squeeze() == action)
            if ok_indices.size(0) >= 1:
                distances = distances[ok_indices]
        closest_idx = torch.argmin(distances)

        data = (
            self.observations[closest_idx],
            self.actions[closest_idx],
            self.next_observations[closest_idx],
            self.means[closest_idx],
            self.sigmas[closest_idx]
        )

        return Transition(*tuple(map(self.to_torch, data))), closest_idx, distances[closest_idx]

    def get_transition(self, transition_index: torch.Tensor):
        data = (
            self.observations[transition_index],
            self.actions[transition_index],
            self.next_observations[transition_index],
            self.means[transition_index],
            self.sigmas[transition_index]
        )

        return Transition(*tuple(map(self.to_torch, data)))

    def to_torch(self, data):
        return torch.Tensor(data)

    def plan(self, latent, horizon: int = 20, k: int = 10, deterministic: bool = False):
        distances = torch.cdist(self.observations[:self._current_idx], latent, p=2)
        indices = torch.topk(-distances, k=k, dim=0).indices.squeeze()
        action_probs = []
        for i in range(horizon):
            step_acts = self.actions[(indices + i) % self._current_idx]
            action_probs.append(torch.bincount(step_acts.flatten().long(), minlength=self.num_actions) / k)
        action_probs = torch.stack(action_probs)
        if deterministic:
            return list(torch.argmax(action_probs, dim=1).numpy())
        else:
            act_distribution = torch.distributions.Categorical(probs=action_probs)
            return act_distribution.sample()

    def plan_distributional(self, mean: torch.Tensor, logvar: torch.Tensor, horizon: int = 20):
        if self._current_idx < self.capacity:
            stored_dists = torch.distributions.Normal(self.means[:self._current_idx], self.sigmas[:self._current_idx] + 1e-8)
            batched_dist = torch.distributions.Normal(mean.repeat(self._current_idx, 1), torch.sqrt(torch.exp(logvar)).repeat(self._current_idx, 1) + 1e-8)
            distances = torch.distributions.kl.kl_divergence(stored_dists, batched_dist).sum(-1)
        else:
            stored_dists = torch.distributions.Normal(self.means, self.sigmas + 1e-8)
            batched_dist = torch.distributions.Normal(mean.repeat(self.means.size(0), 1), torch.sqrt(torch.exp(logvar)).repeat(self.sigmas.size(0), 1) + 1e-8)
            distances = torch.distributions.kl.kl_divergence(stored_dists, batched_dist).sum(-1)

        closest_idx = torch.argmin(distances)

        return [int(self.actions[closest_idx + i % self._current_idx].item()) for i in range(horizon)]