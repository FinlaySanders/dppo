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
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    
    # PREFER hyperparameters
    beta: float = 0.1
    episodes_per_batch: int = 32
    pair_batch_size: int = 128
    update_epochs: int = 10
    
    # Reference policy - maybe do EMA
    reference_update_freq: int = 50


class PolicyNetwork(nn.Module):    
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.net[-1].weight, 0.01)
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, obs: torch.Tensor) -> int:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.sample().item()
    
    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions).mean()
    
    def log_prob_and_entropy(self, obs: torch.Tensor, actions: torch.Tensor):
        """Returns log_prob and entropy, reusing logits computation"""
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
    act_dim = env.action_space.n
    
    # default device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # policy and reference policy
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    reference_policy = PolicyNetwork(obs_dim, act_dim).to(device)
    reference_policy.load_state_dict(policy.state_dict())

    # buffers 
    # TODO: proper buffers for ragged obs and actions
    rewards = torch.zeros((args.episodes_per_batch)).to(device)

    # logging
    global_step = 0
    iteration = 0

    # training loop
    while global_step < args.total_timesteps:
        iteration += 1

        # collect episodes
        episodes = []

        for i in range(args.episodes_per_batch):
            ep = {"obs":[], "acts":[]}
            rewards[i] = 0
            
            obs, _ = env.reset()
            done = False

            while not done:
                with torch.no_grad():
                    action = policy.get_action(torch.tensor(obs, device=device, dtype=torch.float32))
                ep["obs"].append(obs)
                ep["acts"].append(action)

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                rewards[i] += reward

                global_step += 1

            episodes.append(ep)

        print(rewards.mean(), global_step)

        # create preference pairs - skip ties
        idxs, jdxs = torch.triu_indices(len(episodes), len(episodes), offset=1, device=device)

        # Filter out ties (equal rewards)
        gaps = rewards[idxs] - rewards[jdxs]
        keep = gaps.abs() > 1e-6  # Only keep non-ties

        idxs, jdxs, gaps = idxs[keep], jdxs[keep], gaps[keep]

        # Order pairs: winner, loser
        winners = torch.where(gaps > 0, idxs, jdxs)
        losers = torch.where(gaps > 0, jdxs, idxs)

        # Sample batch
        n = min(args.pair_batch_size, len(winners))
        if n > 0:
            sample_idxs = torch.randperm(len(winners), device=device)[:n]
            winners = winners[sample_idxs].tolist()
            losers = losers[sample_idxs].tolist()
        else:
            winners, losers = [], []

        print(len(winners))

        for _ in range(args.update_epochs):
            # compute loss
            losses = []
            for w, l in zip(winners, losers):
                good_obs = torch.tensor(episodes[w]["obs"], device=device)
                good_actions = torch.tensor(episodes[w]["acts"], device=device)
                bad_obs = torch.tensor(episodes[l]["obs"], device=device)
                bad_actions = torch.tensor(episodes[l]["acts"], device=device)

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

        if iteration % args.reference_update_freq == 0:
            reference_policy.load_state_dict(policy.state_dict())

    env.close()

if __name__ == "__main__":
    main()