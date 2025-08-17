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
    exp_name: str = "dppo"
    seed: int = 1
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "dppo"
    wandb_entity: str = None
    
    # Core hyperparameters
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500_000
    learning_rate: float = 3e-3
    num_envs: int = 8
    episodes_per_iteration: int = 32
    
    # DPPO hyperparameters
    beta: float = 0.1
    entropy_coef: float = 0.1
    batch_size: int = 32
    buffer_size: int = 32
    min_episodes_before_training: int = 32
    gradient_steps_per_iteration: int = 1
    percentile: float = 0.2
    
    # Reference policy
    reference_update_freq: int = 50
    
    # Adaptive beta
    adaptive_beta: bool = True


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


class DPPO:    
    def __init__(self, env: gym.Env, args: Args):
        self.env = env
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        
        # Get environment dimensions
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        
        # Initialize policy
        self.policy = PolicyNetwork(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.learning_rate)
        
        # Reference policy for KL regularization
        self.reference_policy = PolicyNetwork(obs_dim, n_actions).to(self.device)
        self.reference_policy.load_state_dict(self.policy.state_dict())
        
        # Episode buffer
        self.episodes = deque(maxlen=args.buffer_size)
        
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name="dppo_entropy_ref50_mean_scaled_ent0.1",
            )
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
        
        self.global_step = 0
        self.iteration = 0
    
    def collect_episode(self) -> Dict:
        obs, _ = self.env.reset(seed=self.args.seed)
        episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "return": 0
        }
        
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
            action = self.policy.get_action(obs_tensor)
            
            episode["observations"].append(obs)
            episode["actions"].append(action)
            
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            episode["rewards"].append(reward)
            episode["return"] += reward
            self.global_step += 1

            if self.args.track:
                wandb.log({"global_step": self.global_step}, step=self.global_step)
        
        episode["observations"] = np.array(episode["observations"], dtype=np.float32)
        episode["actions"] = np.array(episode["actions"], dtype=np.int64)
        episode["rewards"] = np.array(episode["rewards"], dtype=np.float32)
        
        return episode
    
    def normalize_returns(self, episodes: List[Dict]) -> np.ndarray:
        returns = np.array([e["return"] for e in episodes])
        mean = returns.mean()
        advantages = returns - mean
        std = advantages.std() + 1e-8
        normalized = advantages / std
        return normalized
    
    def dpo_step(self, good_episodes: List[Dict], bad_episodes: List[Dict]):
        losses = []
        
        # Compute normalized scores (advantage normalization)
        all_episodes = good_episodes + bad_episodes
        normalized_scores = self.normalize_returns(all_episodes)
        
        # Split back into good/bad
        n_good = len(good_episodes)
        good_scores = normalized_scores[:n_good]
        bad_scores = normalized_scores[n_good:]
        
        # Compute adaptive beta based on score variance
        if self.args.adaptive_beta:
            scores = np.array([e["return"] for e in all_episodes], dtype=np.float32)
            score_std = scores.std() + 1e-8
            adaptive_beta = self.args.beta / score_std
            adaptive_beta = np.clip(adaptive_beta, 0.01, 1.0)
        else:
            adaptive_beta = self.args.beta
        
        # Sample pairs for this gradient step
        n_pairs = min(self.args.batch_size, len(good_episodes), len(bad_episodes))
        for i in range(n_pairs):
            good_idx = random.randint(0, len(good_episodes) - 1)
            bad_idx = random.randint(0, len(bad_episodes) - 1)
            
            good_ep = good_episodes[good_idx]
            bad_ep = bad_episodes[bad_idx]
            
            # Weight based on score difference
            score_diff = good_scores[good_idx] - bad_scores[bad_idx]
            weight = abs(score_diff)
            
            # Convert to tensors
            good_obs = torch.tensor(good_ep["observations"], device=self.device)
            good_actions = torch.tensor(good_ep["actions"], device=self.device)
            bad_obs = torch.tensor(bad_ep["observations"], device=self.device)
            bad_actions = torch.tensor(bad_ep["actions"], device=self.device)
            
            # Compute log probabilities
            logp_good = self.policy.log_prob(good_obs, good_actions)
            logp_bad = self.policy.log_prob(bad_obs, bad_actions)
            
            with torch.no_grad():
                ref_logp_good = self.reference_policy.log_prob(good_obs, good_actions)
                ref_logp_bad = self.reference_policy.log_prob(bad_obs, bad_actions)
            
            Lg = len(good_ep["actions"])
            Lb = len(bad_ep["actions"])
            beta_eff = adaptive_beta * (Lg + Lb) / 2.0

            # DPO with reference policy and beta
            logits = beta_eff * (
                (logp_good - ref_logp_good) - 
                (logp_bad - ref_logp_bad)
            )
            
            # Bradley-Terry loss with weighting
            loss = -F.logsigmoid(logits) * weight

            entropy = 0.5 * (
                Categorical(logits=self.policy(good_obs)).entropy().mean() +
                Categorical(logits=self.policy(bad_obs)).entropy().mean()
            )
            loss = loss - self.args.entropy_coef * entropy

            losses.append(loss)
        
        # Optimize
        if losses:
            total_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            if self.args.track:
                wandb.log({
                    "loss/dpo": total_loss.item(),
                    "debug/adaptive_beta": adaptive_beta,
                }, step=self.global_step)
            
            return total_loss.item()
        return 0.0
    
    def train_step(self):
        self.iteration += 1
        
        # Collect new episodes
        new_episodes = []
        for _ in range(self.args.episodes_per_iteration):
            episode = self.collect_episode()
            new_episodes.append(episode)
            self.episodes.append(episode)
            
            if self.args.track:
                wandb.log({
                    "charts/episodic_return": episode["return"],
                    "charts/episode_length": len(episode["actions"]),
                }, step=self.global_step)
        
        # Wait until we have enough episodes
        if len(self.episodes) < self.args.min_episodes_before_training:
            return
                
        # Rank episodes by return
        sorted_episodes = sorted(self.episodes, key=lambda e: e["return"])
        n = len(sorted_episodes)
                
        # Select good and bad episodes based on percentiles
        good_threshold = int((1-self.args.percentile) * n)
        bad_threshold = int(self.args.percentile * n)
        
        good_episodes = sorted_episodes[good_threshold:]
        bad_episodes = sorted_episodes[:bad_threshold]
        
        # Perform gradient steps
        total_loss = 0
        for _ in range(self.args.gradient_steps_per_iteration):
            loss = self.dpo_step(good_episodes, bad_episodes)
            total_loss += loss
                
        # Update reference policy periodically
        if self.iteration % self.args.reference_update_freq == 0:
            self.reference_policy.load_state_dict(self.policy.state_dict())
    
    def train(self):
        start_time = time.time()
        
        while self.global_step < self.args.total_timesteps:
            self.train_step()
            
            if self.args.track and self.iteration % 10 == 0:
                sps = int(self.global_step / (time.time() - start_time))
                wandb.log({"charts/SPS": sps}, step=self.global_step)

def main():
    args = tyro.cli(Args)
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = gym.make(args.env_id)
    
    # Create and train agent
    agent = DPPO(env, args)
    agent.train()
    
    env.close()


if __name__ == "__main__":
    main()