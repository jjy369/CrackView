"""Single-step Gaussian PPO. Each rollout has 32 contexts and 320 labeled actions."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Independent


def mlp(widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim=4777, actor=(4096,2048,1024,512), critic=(2048,1024,512)):
        super().__init__()
        self.actor = mlp([obs_dim, *actor, action_dim])
        self.critic = mlp([obs_dim, *critic, 1])
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.15))

    def distribution(self, obs):
        return Independent(Normal(self.actor(obs), self.log_std.exp()), 1)

    @torch.no_grad()
    def draw(self, obs):
        dist = self.distribution(obs)
        actions = dist.sample((10,))
        return actions, dist.log_prob(actions), self.critic(obs).squeeze(-1)


def collect(env, policy, contexts=32):
    device = next(policy.parameters()).device
    rows = []
    for _ in range(contexts):
        sid = env._sample_scene()
        aux = str(env.rng.choice([s for s in env.all_scene_ids if s != sid]))
        obs = torch.as_tensor(env._get_scene_obs(sid), device=device)
        actions, old_logp, value = policy.draw(obs)
        raw = actions.cpu().numpy()
        rewards = (env.rewards(sid, raw) + .25 * env.rewards(aux, raw)) / 1.25
        returns = torch.as_tensor(rewards, dtype=obs.dtype, device=device)
        rows.append((obs.expand(10,-1), actions, old_logp, returns, returns - value))
        env.episode_idx += 1
    return tuple(torch.cat(column) for column in zip(*rows))


def update(policy, optimizer, batch, entropy_coef):
    obs, actions, old, returns, advantages = batch
    for _ in range(10):
        for ids in torch.randperm(len(obs), device=obs.device).split(32):
            dist = policy.distribution(obs[ids])
            log_ratio = dist.log_prob(actions[ids]) - old[ids]
            ratio = log_ratio.exp()
            if (ratio - 1 - log_ratio).mean().detach().item() > .020:
                return
            adv = advantages[ids]
            surrogate = torch.minimum(ratio * adv, ratio.clamp(.9, 1.1) * adv)
            value_loss = (policy.critic(obs[ids]).flatten() - returns[ids]).square().mean()
            loss = -surrogate.mean() + .45 * value_loss - entropy_coef * dist.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
