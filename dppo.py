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
    learning_rate: float = 3e-4
    episodes_per_iteration: int = 32
    
    # DPPO hyperparameters
    beta: float = 0.1
    entropy_coef: float = 0.01
    batch_size: int = 32
    min_episodes_before_training: int = 32
    gradient_steps_per_iteration: int = 25
    percentile: float = 0.2
    
    # Reference policy
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
                
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name="dppo_yes",
            )
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
        
        self.global_step = 0
        self.iteration = 0
    
    def collect_episode(self) -> Dict:
        obs, _ = self.env.reset()
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
    
    def dppo_step(self, good_episodes: List[Dict], bad_episodes: List[Dict]):
        losses = []
        
        # Compute normalized scores (advantage normalization)
        all_episodes = good_episodes + bad_episodes
        normalized_scores = self.normalize_returns(all_episodes)
        
        # Split back into good/bad
        n_good = len(good_episodes)
        good_scores = normalized_scores[:n_good]
        bad_scores = normalized_scores[n_good:]
        
        # Sample pairs for this gradient step
        n_pairs = min(self.args.batch_size, len(good_episodes), len(bad_episodes))
        
        # First pass: collect all signal magnitudes to compute adaptive beta
        signal_magnitudes = []
        sampled_pairs = []
        
        for _ in range(n_pairs):
            good_idx = random.randint(0, len(good_episodes) - 1)
            bad_idx = random.randint(0, len(bad_episodes) - 1)
            
            good_ep = good_episodes[good_idx]
            bad_ep = bad_episodes[bad_idx]
            
            # Store the sampled pair for later use
            sampled_pairs.append((good_idx, bad_idx, good_ep, bad_ep))
            
            # Convert to tensors
            good_obs = torch.tensor(good_ep["observations"], device=self.device)
            good_actions = torch.tensor(good_ep["actions"], device=self.device)
            bad_obs = torch.tensor(bad_ep["observations"], device=self.device)
            bad_actions = torch.tensor(bad_ep["actions"], device=self.device)
            
            # Compute signal magnitude without gradients
            with torch.no_grad():
                logp_good = self.policy.log_prob(good_obs, good_actions)
                logp_bad = self.policy.log_prob(bad_obs, bad_actions)
                ref_logp_good = self.reference_policy.log_prob(good_obs, good_actions)
                ref_logp_bad = self.reference_policy.log_prob(bad_obs, bad_actions)
                
                signal_good = (logp_good - ref_logp_good).abs()
                signal_bad = (logp_bad - ref_logp_bad).abs()
                signal_magnitudes.append((signal_good + signal_bad) / 2)
        
        # Compute adaptive beta once for all pairs
        if signal_magnitudes:
            avg_signal = torch.stack(signal_magnitudes).mean().item()
            target_signal = 0.1  # Target signal magnitude - tune this
            
            # Scale by training progress (0.1 -> 1.0)
            progress = min(1.0, 0.1 + 0.9 * self.global_step / self.args.total_timesteps)
            signal_multiplier = target_signal / (avg_signal + 0.01)
            adaptive_beta = np.clip(self.args.beta * signal_multiplier * progress, 0.01, 1.0)
        else:
            adaptive_beta = self.args.beta
        
        # Second pass: compute losses with consistent adaptive beta
        for good_idx, bad_idx, good_ep, bad_ep in sampled_pairs:
            # Weight based on score difference
            score_diff = good_scores[good_idx] - bad_scores[bad_idx]
            weight = abs(score_diff)
            
            # Convert to tensors
            good_obs = torch.tensor(good_ep["observations"], device=self.device)
            good_actions = torch.tensor(good_ep["actions"], device=self.device)
            bad_obs = torch.tensor(bad_ep["observations"], device=self.device)
            bad_actions = torch.tensor(bad_ep["actions"], device=self.device)
            
            # Compute log probabilities and entropy (with gradients this time)
            logp_good, entropy_good = self.policy.log_prob_and_entropy(good_obs, good_actions)
            logp_bad, entropy_bad = self.policy.log_prob_and_entropy(bad_obs, bad_actions)
            
            # Compute reference log probabilities without gradients
            with torch.no_grad():
                ref_logp_good = self.reference_policy.log_prob(good_obs, good_actions)
                ref_logp_bad = self.reference_policy.log_prob(bad_obs, bad_actions)
            
            # DPPO with reference policy and beta
            # No length scaling needed since log_prob already averages
            logits = adaptive_beta * (
                (logp_good - ref_logp_good) - 
                (logp_bad - ref_logp_bad)
            )
            
            # Bradley-Terry loss with weighting
            loss = -F.logsigmoid(logits) * weight
            
            # Add entropy bonus (average of both episodes)
            entropy = 0.5 * (entropy_good + entropy_bad)
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
                    "loss/dppo": total_loss.item(),
                    "debug/adaptive_beta": adaptive_beta,
                    "debug/avg_signal": avg_signal if signal_magnitudes else 0,
                    "debug/progress": progress if signal_magnitudes else 0,
                }, step=self.global_step)
            
            return total_loss.item()
        else:
            print("Warning: No losses computed in dppo_step")
            return 0.0
    
    def train_step(self):
        self.iteration += 1
        
        # Collect new episodes
        episodes = []
        for _ in range(self.args.episodes_per_iteration):
            episode = self.collect_episode()
            episodes.append(episode)
            
            if self.args.track:
                wandb.log({
                    "charts/episodic_return": episode["return"],
                    "charts/episode_length": len(episode["actions"]),
                }, step=self.global_step)
        
        # Wait until we have enough episodes
        if len(episodes) < self.args.min_episodes_before_training:
            return
                
        # Rank episodes by return
        sorted_episodes = sorted(episodes, key=lambda e: e["return"])
        n = len(sorted_episodes)
                
        # Select good and bad episodes based on percentiles
        good_threshold = int((1-self.args.percentile) * n)
        bad_threshold = int(self.args.percentile * n)
        
        # Ensure we have at least one episode in each category
        good_threshold = max(n - 1, good_threshold)  # At least 1 good episode
        bad_threshold = min(1, bad_threshold)  # At least 1 bad episode
        
        good_episodes = sorted_episodes[good_threshold:]
        bad_episodes = sorted_episodes[:bad_threshold]
        
        # Validate we have episodes in both categories
        if not good_episodes or not bad_episodes:
            print(f"Warning: Skipping training - good: {len(good_episodes)}, bad: {len(bad_episodes)}")
            return
        
        # Perform gradient steps
        for _ in range(self.args.gradient_steps_per_iteration):
            self.dppo_step(good_episodes, bad_episodes)
                
        # Update reference policy periodically
        if self.iteration % self.args.reference_update_freq == 0:
            self.reference_policy.load_state_dict(self.policy.state_dict())
            if self.args.track:
                wandb.log({"debug/reference_update": 1}, step=self.global_step)
    
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
    env.reset(seed=args.seed)
    
    # Create and train agent
    agent = DPPO(env, args)
    agent.train()
    
    env.close()


if __name__ == "__main__":
    main()