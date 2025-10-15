import os
import pdb
import torch
import scipy

from tqdm import trange
from functools import partial
from collections import defaultdict

from torch.distributions import Normal, kl
from torch.distributions.kl import kl_divergence

from utils_planet import *
from memory import *
from rssm_model import *
from rssm_policy import *
from rollout_generator import RolloutGenerator, PyStkRolloutGenerator

import gymnasium as gym
import super_tux_env


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def train(memory, rssm, optimizer, device, N=32, H=50, beta=1.0, grads=False):
    """
    Training implementation as indicated in:
    Learning Latent Dynamics for Planning from Pixels
    arXiv:1811.04551

    (a.) The Standard Variational Bound Method
        using only single step predictions.
    """
    free_nats = torch.ones(1, device=device) * 3.0
    batch = memory.sample(N, H, time_first=True)
    x, u, r, t = [torch.tensor(x).float().to(device) for x in batch]
    preprocess_img(x, depth=5)
    e_t = bottle(rssm.encoder, x)
    h_t, s_t = rssm.get_init_state(e_t[0])
    kl_loss, rc_loss, re_loss = 0, 0, 0
    states, priors, posteriors, posterior_samples = [], [], [], []
    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        states.append(h_t)
        priors.append(rssm.state_prior(h_t))
        posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
        posterior_samples.append(Normal(*posteriors[-1]).rsample())
        s_t = posterior_samples[-1]
    prior_dist = Normal(*map(torch.stack, zip(*priors)))
    posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
    states, posterior_samples = map(torch.stack, (states, posterior_samples))
    rec_loss = F.mse_loss(
        bottle(rssm.decoder, states, posterior_samples), x[1:],
        reduction='none'
    ).sum((2, 3, 4)).mean()
    kld_loss = torch.max(
        kl_divergence(posterior_dist, prior_dist).sum(-1),
        free_nats
    ).mean()
    rew_loss = F.mse_loss(
        bottle(rssm.pred_reward, states, posterior_samples), r
    )
    optimizer.zero_grad()
    nn.utils.clip_grad_norm_(rssm.parameters(), 1000., norm_type=2)
    (beta * kld_loss + rec_loss + rew_loss).backward()
    optimizer.step()
    metrics = {
        'losses': {
            'kl': kld_loss.item(),
            'reconstruction': rec_loss.item(),
            'reward_pred': rew_loss.item()
        },
    }
    if grads:
        metrics['grad_norms'] = {
            k: 0 if v.grad is None else v.grad.norm().item()
            for k, v in rssm.named_parameters()
        }
    return metrics


def main():
    track_name = 'fortmagma'
    train_model = False
    res_dir = f'planet_model/{track_name}'
    model_name = "ckpt_25"
    num_eval_episodes = 1
    max_episode_steps = 2500
    epochs = 25 # Selected as comparable exposure to search-based transitions
    checkpoint_frequency = int(epochs / 5)

    env = gym.make('SuperTuxEnv-v0', track_name=track_name, reward_scheme="nodes", use_time_reward=False)
    env = TuxKartTorchImageEnvWrapper(env, bit_depth=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rssm_model = RecurrentStateSpaceModel(env.action_size).to(device)
    optimizer = torch.optim.Adam(rssm_model.parameters(), lr=1e-3, eps=1e-4)
    policy = RSSMTuxKartPolicy(
        rssm_model,
        planning_horizon=20,
        num_candidates=1000,
        num_iterations=10,
        top_candidates=100,
        device=device
    )
    rollout_gen = PyStkRolloutGenerator(
        env,
        device,
        policy=policy,
        episode_gen=lambda: Episode(partial(postprocess_img, depth=5)),
        max_episode_steps=max_episode_steps,
    )
    if train_model:
        mem = Memory(100)
        mem.append(rollout_gen.rollout_n(1, random_policy=True))
        summary = TensorBoardMetrics(f'{res_dir}/')
        for i in trange(epochs, desc='Epoch', leave=False):
            metrics = {}
            for _ in trange(150, desc='Iter ', leave=False):
                train_metrics = train(mem, rssm_model.train(), optimizer, device)
                for k, v in flatten_dict(train_metrics).items():
                    if k not in metrics.keys():
                        metrics[k] = []
                    metrics[k].append(v)
                    metrics[f'{k}_mean'] = np.array(v).mean()

            summary.update(metrics)
            mem.append(rollout_gen.rollout_once(explore=True))
            eval_episode, eval_frames, eval_metrics = rollout_gen.rollout_eval()
            mem.append(eval_episode)
            save_video(eval_frames, res_dir, f'vid_{i + 1}')
            summary.update(eval_metrics)

            if (i + 1) % checkpoint_frequency == 0:
                torch.save(rssm_model.state_dict(), f'{res_dir}/ckpt_{i + 1}.pth')

    else:
        rssm_model.load_state_dict(torch.load(f'{res_dir}/{model_name}.pth'))
        rssm_model.eval()

    os.makedirs(f"{res_dir}/evaluation", exist_ok=True)
    with open(f"{res_dir}/evaluation/reward_evaluation.csv", "a+") as f:
        f.write("mean,std,ci_95_low,ci_95_high,ci_99_low,ci_99_high\n")

    rewards = []
    progress_bar = trange(num_eval_episodes, desc="Evaluating model")
    for i in progress_bar:
        eval_episode, eval_frames, eval_metrics = rollout_gen.rollout_eval()
        rewards.append(eval_metrics['eval/episode_reward'])
        save_video(eval_frames, f"{res_dir}/evaluation", f'vid_{i + 1}')
        if i % 10 == 0:
            progress_bar.set_description(f"Avg. reward: {np.mean(rewards):.5f} +/- {np.std(rewards):.5f}")

    mean, lower_95, upper_95 = mean_confidence_interval(rewards, confidence=0.95)
    _, lower_99, upper_99 = mean_confidence_interval(rewards, confidence=0.99)

    with open(f"{res_dir}/evaluation/reward_evaluation.csv", "a+") as f:
        f.write(f"{float(np.mean(rewards)):.4f},{float(np.std(rewards)):.4f},{lower_95:.4f},{upper_95:.4f},{lower_99:.4f},{upper_99:.4f}\n")


if __name__ == '__main__':
    main()
