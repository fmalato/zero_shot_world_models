import torch
import torch.nn as nn
import torch.nn.functional as F


class NextStepPredictionSSM(nn.Module):
    def __init__(self, encoded_latent_size: int, action_size: int, hidden_state_size: int, context_size: int = 4):
        super(NextStepPredictionSSM, self).__init__()
        # Deterministic state
        self.obs_action_embedding = nn.Linear(context_size * encoded_latent_size + action_size, hidden_state_size)
        self.gru = nn.GRUCell(hidden_state_size, hidden_state_size)
        # Stochastic prior
        self.stochastic_state_prior_embed = nn.Linear(hidden_state_size, hidden_state_size)
        self.stochastic_state_prior_mean = nn.Linear(hidden_state_size, encoded_latent_size)
        self.stochastic_state_prior_logvar = nn.Linear(hidden_state_size, encoded_latent_size)
        # Stochastic posterior
        self.state_posterior_embed = nn.Linear(hidden_state_size + encoded_latent_size, hidden_state_size)
        self.state_posterior_mean = nn.Linear(hidden_state_size, encoded_latent_size)
        self.state_posterior_logvar = nn.Linear(hidden_state_size, encoded_latent_size)

        self.hidden_state_size = hidden_state_size
        self.action_size = action_size
        self.context_size = context_size

    def forward(self, h_t_prev, s_t_prev, a_t_prev, device):
        # Get the deterministic state h_t = f(h_t-1, s_t-1, a_t-1)
        one_hot_action = F.softmax(F.one_hot(a_t_prev.long(), num_classes=self.action_size).squeeze(1).float(), dim=-1).to(device)
        s_t_prev = s_t_prev.view(s_t_prev.size(0), -1).to(device)
        obs_act_embed = torch.cat([s_t_prev, one_hot_action], dim=-1)
        encoded_obs_act = self.obs_action_embedding(obs_act_embed)
        h_t = self.gru(encoded_obs_act, h_t_prev)
        # Get the stochastic state p(s_t | h_t)
        s_t = self.stochastic_state_prior_embed(h_t)
        s_t_mean = self.stochastic_state_prior_mean(s_t)
        s_t_logvar = self.stochastic_state_prior_logvar(s_t)
        # Sample the state
        s_t_sample = torch.distributions.Normal(s_t_mean, torch.exp(s_t_logvar)).rsample()
        # Get the next observation p(o_t | h_t, s_t)
        complete_state = torch.cat([h_t, s_t_sample], dim=-1)
        z_t = self.state_posterior_embed(complete_state)
        z_t_mean = self.state_posterior_mean(z_t)
        z_t_logvar = self.state_posterior_logvar(z_t)
        # Sample observation
        o_t = torch.distributions.Normal(z_t_mean, torch.exp(z_t_logvar)).rsample()

        return o_t, h_t, s_t_mean, s_t_logvar, z_t_mean, z_t_logvar

    def get_previous_hidden(self, o, a, device):
        with torch.no_grad():
            h_t_prev = torch.zeros(size=(a.size(0), self.hidden_state_size)).to(device)
            obs_act_embed = torch.cat([o.view(o.size(0), -1), F.softmax(F.one_hot(a.long(), num_classes=self.action_size).squeeze(1).float(), dim=-1)], dim=-1).to(device)
            encoded_obs_act = self.obs_action_embedding(obs_act_embed)
            h_t = self.gru(encoded_obs_act, h_t_prev)

        return h_t

    def get_context_prior(self, h, s, a, device):
        stats = {
            "prior_mean": None,
            "prior_logvar": None,
            "posterior_mean": [],
            "posterior_logvar": []
        }
        with torch.no_grad():
            one_hot_action = F.softmax(F.one_hot(a.long(), num_classes=self.action_size).squeeze(1).float(), dim=-1).to(device)
            s_t_prev = s.view(s.size(0), -1).to(device)
            obs_act_embed = torch.cat([s_t_prev, one_hot_action], dim=-1)
            encoded_obs_act = self.obs_action_embedding(obs_act_embed)
            h_t = self.gru(encoded_obs_act, h)
            # Get the stochastic state p(s_t | h_t)
            s_t = self.stochastic_state_prior_embed(h_t)
            s_t_mean = self.stochastic_state_prior_mean(s_t)
            s_t_logvar = self.stochastic_state_prior_logvar(s_t)
            # Sample the state
            s_t_sample = torch.distributions.Normal(s_t_mean, torch.exp(s_t_logvar)).rsample()

            stats["posterior_mean"] = s_t_mean.unsqueeze(0)
            stats["posterior_logvar"] = s_t_logvar.unsqueeze(0)

        return s_t_sample, stats

    def predict(self, h_t, o_t, a_t, device, horizon=1):
        time_sequence = []
        stats = {
            "prior_mean": [],
            "prior_logvar": [],
            "posterior_mean": [],
            "posterior_logvar": []
        }
        obs_buffer = [o_t[:, i, :].unsqueeze(1).to(device) for i in range(o_t.size(1))]
        with torch.no_grad():
            for t in range(horizon):
                o_next, h_t_next, prior_mean, prior_logvar, posterior_mean, posterior_logvar = self.forward(h_t, o_t, a_t[:, t], device)
                time_sequence.append(o_next)
                obs_buffer.pop(0)
                obs_buffer.append(o_next.unsqueeze(1))
                o_t = torch.cat(obs_buffer, dim=1)
                h_t = h_t_next
                stats["prior_mean"].append(prior_mean)
                stats["prior_logvar"].append(prior_logvar)
                stats["posterior_mean"].append(posterior_mean)
                stats["posterior_logvar"].append(posterior_logvar)

        return time_sequence, stats


class NextStepPredictionSSMMineRL(nn.Module):
    def __init__(self, encoded_latent_size: int, action_size: int, hidden_state_size: int, context_size: int = 4, is_minerl: bool = False):
        super(NextStepPredictionSSMMineRL, self).__init__()
        # Deterministic state
        self.obs_action_embedding = nn.Linear(context_size * encoded_latent_size + action_size, hidden_state_size)
        self.gru = nn.GRUCell(hidden_state_size, hidden_state_size)
        # Stochastic prior
        self.stochastic_state_prior_embed = nn.Linear(hidden_state_size, hidden_state_size)
        self.stochastic_state_prior_mean = nn.Linear(hidden_state_size, encoded_latent_size)
        self.stochastic_state_prior_logvar = nn.Linear(hidden_state_size, encoded_latent_size)
        # Stochastic posterior
        self.state_posterior_embed = nn.Linear(hidden_state_size + encoded_latent_size, hidden_state_size)
        self.state_posterior_mean = nn.Linear(hidden_state_size, encoded_latent_size)
        self.state_posterior_logvar = nn.Linear(hidden_state_size, encoded_latent_size)

        self.hidden_state_size = hidden_state_size
        self.action_size = action_size
        self.context_size = context_size
        self.is_minerl = is_minerl

    def forward(self, h_t_prev, s_t_prev, a_t_prev, device):
        # Get the deterministic state h_t = f(h_t-1, s_t-1, a_t-1)
        if not self.is_minerl:
            one_hot_action = F.softmax(F.one_hot(a_t_prev.long(), num_classes=self.action_size).squeeze(1).float(), dim=-1).to(device)
        else:
            one_hot_action = a_t_prev.float().to(device)
        s_t_prev = s_t_prev.view(s_t_prev.size(0), -1).to(device)
        obs_act_embed = torch.cat([s_t_prev, one_hot_action], dim=-1)
        encoded_obs_act = self.obs_action_embedding(obs_act_embed)
        h_t = self.gru(encoded_obs_act, h_t_prev)
        # Get the stochastic state p(s_t | h_t)
        s_t = self.stochastic_state_prior_embed(h_t)
        s_t_mean = self.stochastic_state_prior_mean(s_t)
        s_t_logvar = self.stochastic_state_prior_logvar(s_t)
        # Sample the state
        s_t_sample = torch.distributions.Normal(s_t_mean, torch.exp(s_t_logvar)).rsample()
        # Get the next observation p(o_t | h_t, s_t)
        complete_state = torch.cat([h_t, s_t_sample], dim=-1)
        z_t = self.state_posterior_embed(complete_state)
        z_t_mean = self.state_posterior_mean(z_t)
        z_t_logvar = self.state_posterior_logvar(z_t)
        # Sample observation
        o_t = torch.distributions.Normal(z_t_mean, torch.exp(z_t_logvar)).rsample()

        return o_t, h_t, s_t_mean, s_t_logvar, z_t_mean, z_t_logvar

    def get_previous_hidden(self, o, a, device):
        with torch.no_grad():
            h_t_prev = torch.zeros(size=(a.size(0), self.hidden_state_size)).to(device)
            if not self.is_minerl:
                acts = F.softmax(F.one_hot(a.long(), num_classes=self.action_size).squeeze(1).float(), dim=-1).to(device)
            else:
                acts = a.float().to(device)
            obs_act_embed = torch.cat([o.view(o.size(0), -1), acts], dim=-1).to(device)
            encoded_obs_act = self.obs_action_embedding(obs_act_embed)
            h_t = self.gru(encoded_obs_act, h_t_prev)

        return h_t

    def get_context_prior(self, h, s, a, device):
        stats = {
            "prior_mean": None,
            "prior_logvar": None,
            "posterior_mean": [],
            "posterior_logvar": []
        }
        with torch.no_grad():
            if not self.is_minerl:
                acts = F.softmax(F.one_hot(a.long(), num_classes=self.action_size).squeeze(1).float(), dim=-1).to(device)
            else:
                acts = a.float().to(device)
            s_t_prev = s.view(s.size(0), -1).to(device)
            obs_act_embed = torch.cat([s_t_prev, acts], dim=-1)
            encoded_obs_act = self.obs_action_embedding(obs_act_embed)
            h_t = self.gru(encoded_obs_act, h)
            # Get the stochastic state p(s_t | h_t)
            s_t = self.stochastic_state_prior_embed(h_t)
            s_t_mean = self.stochastic_state_prior_mean(s_t)
            s_t_logvar = self.stochastic_state_prior_logvar(s_t)
            # Sample the state
            s_t_sample = torch.distributions.Normal(s_t_mean, torch.exp(s_t_logvar)).rsample()

            stats["posterior_mean"] = s_t_mean.unsqueeze(0)
            stats["posterior_logvar"] = s_t_logvar.unsqueeze(0)

        return s_t_sample, stats

    def predict(self, h_t, o_t, a_t, device, horizon=1):
        time_sequence = []
        stats = {
            "prior_mean": [],
            "prior_logvar": [],
            "posterior_mean": [],
            "posterior_logvar": []
        }
        obs_buffer = [o_t[:, i, :].unsqueeze(1).to(device) for i in range(o_t.size(1))]
        with torch.no_grad():
            for t in range(horizon):
                o_next, h_t_next, prior_mean, prior_logvar, posterior_mean, posterior_logvar = self.forward(h_t, o_t, a_t[:, t], device)
                time_sequence.append(o_next)
                obs_buffer.pop(0)
                obs_buffer.append(o_next.unsqueeze(1))
                o_t = torch.cat(obs_buffer, dim=1)
                h_t = h_t_next
                stats["prior_mean"].append(prior_mean)
                stats["prior_logvar"].append(prior_logvar)
                stats["posterior_mean"].append(posterior_mean)
                stats["posterior_logvar"].append(posterior_logvar)

        return time_sequence, stats
