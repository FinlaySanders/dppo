# PREFER: Preference Based Reinforcement Learning from Environment Rewards

import random
import time
from dataclasses import dataclass
from collections import deque
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import wandb

@dataclass
class Args:
    exp_name: str = "prefer"
    seed: int = 1
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "prefer"
    wandb_entity: str = None
    
    # Core hyperparameters
    env_id: str = "CartPole-v1"
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    
    # PREFER hyperparameters
    beta: float = 1.0
    episode_buffer_size: int = 32
    episodes_per_batch: int = 32
    pair_batch_size: int = 128
    update_epochs: int = 10
    
    # Reference policy
    tau: float = 0.001


class PolicyNetwork(nn.Module):    
    def __init__(self, obs_dim: int, act_dim: int, continuous: bool = False):
        super().__init__()
        self.continuous = continuous
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        
        if continuous:
            self.mean_head = nn.Linear(64, act_dim)
            self.log_std = nn.Parameter(torch.zeros(1, act_dim))
        else:
            self.logits_head = nn.Linear(64, act_dim)
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        if continuous:
            nn.init.orthogonal_(self.mean_head.weight, 0.01)
            nn.init.constant_(self.mean_head.bias, 0.0)
        else:
            nn.init.orthogonal_(self.logits_head.weight, 0.01)
            nn.init.constant_(self.logits_head.bias, 0.0)
    
    def forward(self, x):
        features = self.net(x)
        if self.continuous:
            mean = self.mean_head(features)
            std = self.log_std.exp().expand_as(mean)
            return mean, std
        else:
            return self.logits_head(features)
    
    def get_action(self, obs: torch.Tensor):
        if self.continuous:
            # Ensure obs is batched
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            mean, std = self.forward(obs)
            dist = Normal(mean, std)
            action = dist.sample()
            return action.squeeze(0).cpu().numpy()
        else:
            logits = self.forward(obs)
            dist = Categorical(logits=logits)
            return dist.sample().item()
    
    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.continuous:
            mean, std = self.forward(obs)
            dist = Normal(mean, std)
            return dist.log_prob(actions).sum(dim=-1).mean()
        else:
            logits = self.forward(obs)
            dist = Categorical(logits=logits)
            return dist.log_prob(actions).mean()
    
    def log_prob_and_entropy(self, obs: torch.Tensor, actions: torch.Tensor):
        if self.continuous:
            mean, std = self.forward(obs)
            dist = Normal(mean, std)
            return dist.log_prob(actions).sum(dim=-1).mean(), dist.entropy().sum(dim=-1).mean()
        else:
            logits = self.forward(obs)
            dist = Categorical(logits=logits)
            return dist.log_prob(actions).mean(), dist.entropy().mean()


def main():
    args = tyro.cli(Args)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create env
    env = gym.make(args.env_id)
    env.reset(seed=args.seed)

    # env dimensions
    obs_dim = env.observation_space.shape[0]
    
    # Check if action space is continuous or discrete
    if isinstance(env.action_space, gym.spaces.Box):
        continuous = True
        act_dim = env.action_space.shape[0]
    else:
        continuous = False
        act_dim = env.action_space.n
    
    # default device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # policy and reference policy
    policy = PolicyNetwork(obs_dim, act_dim, continuous).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    reference_policy = PolicyNetwork(obs_dim, act_dim, continuous).to(device)
    reference_policy.load_state_dict(policy.state_dict())

    # buffers 
    # TODO: make proper buffers for obs/acts
    episode_buffer = deque(maxlen=args.episode_buffer_size)
    reward_buffer = deque(maxlen=args.episode_buffer_size)

    # logging
    global_step = 0
    iteration = 0

    # training loop
    while global_step < args.total_timesteps:
        iteration += 1

        # collect episodes
        tot = 0
        for i in range(args.episodes_per_batch):
            ep = {"obs":[], "acts":[]}
            rew = 0
            
            obs, _ = env.reset()
            done = False

            while not done:
                with torch.no_grad():
                    action = policy.get_action(torch.tensor(obs, device=device, dtype=torch.float32))
                
                ep["obs"].append(obs)
                ep["acts"].append(action)

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                rew += reward

                global_step += 1

            episode_buffer.append(ep)
            reward_buffer.append(rew)
            tot += rew

        print(tot / args.episodes_per_batch, global_step)

        # create preference pairs - skip ties
        idxs, jdxs = torch.triu_indices(len(episode_buffer), len(episode_buffer), offset=1, device=device)
        rewards = torch.tensor(list(reward_buffer), device=device, dtype=torch.float32)
        gaps = rewards[idxs] - rewards[jdxs]
        keep = gaps.abs() > 1e-6
        idxs, jdxs, gaps = idxs[keep], jdxs[keep], gaps[keep]
        winners = torch.where(gaps > 0, idxs, jdxs)
        losers = torch.where(gaps > 0, jdxs, idxs)

        # Sample batch
        n = min(args.pair_batch_size, len(winners))
        sample_idxs = torch.randperm(len(winners), device=device)[:n]
        winners = winners[sample_idxs].tolist()
        losers = losers[sample_idxs].tolist()

        for _ in range(args.update_epochs):
            # compute loss
            losses = []
            for w, l in zip(winners, losers):
                good_obs = torch.tensor(np.array(episode_buffer[w]["obs"]), device=device, dtype=torch.float32)
                good_actions = torch.tensor(np.array(episode_buffer[w]["acts"]), device=device, dtype=torch.float32)
                bad_obs = torch.tensor(np.array(episode_buffer[l]["obs"]), device=device, dtype=torch.float32)
                bad_actions = torch.tensor(np.array(episode_buffer[l]["acts"]), device=device, dtype=torch.float32)

                logp_good, entropy_good = policy.log_prob_and_entropy(good_obs, good_actions)
                logp_bad, entropy_bad = policy.log_prob_and_entropy(bad_obs, bad_actions)

                with torch.no_grad():
                    ref_logp_good = reference_policy.log_prob(good_obs, good_actions)
                    ref_logp_bad = reference_policy.log_prob(bad_obs, bad_actions)

                logits = args.beta * (
                    (logp_good - ref_logp_good) - 
                    (logp_bad - ref_logp_bad)
                )

                loss = -F.logsigmoid(logits)
                                
                losses.append(loss)
            
            # no preference to learn from!
            if not losses:
                continue

            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        for param, ref_param in zip(policy.parameters(), reference_policy.parameters()):
            ref_param.data.copy_(args.tau * param.data + (1 - args.tau) * ref_param.data)

    env.close()

if __name__ == "__main__":
    main()