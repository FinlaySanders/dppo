import os
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
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = "pure_dpo_rl_normalized"
    seed: int = 18
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    
    # Core hyperparameters
    env_id: str = "CartPole-v1"
    total_timesteps: int = 10000000
    learning_rate: float = 3e-3
    num_envs: int = 8
    episodes_per_iteration: int = 32
    
    # DPO hyperparameters
    beta: float = 0.1
    batch_size: int = 32
    buffer_size: int = 32
    percentile_gap: float = 0.1
    min_episodes_before_training: int = 0
    gradient_steps_per_iteration: int = 1
    
    # Reference policy
    use_reference: bool = True
    reference_update_freq: int = 10
    
    # Normalization options
    normalize_returns: bool = True  # Enable return normalization
    normalization_type: str = "advantage"  # Options: "zscore", "rank", "adaptive", "minmax", "percentage" - TODO breaks with negative returns, "advantage"
    adaptive_beta: bool = True  # Scale beta based on return variance or proximity to max
    temperature_scaling: float = 1.0  # Temperature for softmax-based normalization
    max_episode_return: float = 500.0  # Maximum possible return (auto-set for known envs)


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
        return dist.log_prob(actions).sum()


class MinimalDPOAgent:    
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
        self.reference_policy = None
        if args.use_reference:
            self.reference_policy = PolicyNetwork(obs_dim, n_actions).to(self.device)
            self.reference_policy.load_state_dict(self.policy.state_dict())
        
        # Episode buffer
        self.episodes = deque(maxlen=args.buffer_size)
        
        # Track return statistics for adaptive normalization
        self.return_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": 0.0,
            "max": 1.0
        }
        
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=f"{args.exp_name}_{args.seed}_{int(time.time())}",
                monitor_gym=True,
                save_code=True,
            )

        # Logging
        self.writer = None
        if args.track:
            run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
            self.writer = SummaryWriter(f"runs/{run_name}")
        
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
        
        episode["observations"] = np.array(episode["observations"], dtype=np.float32)
        episode["actions"] = np.array(episode["actions"], dtype=np.int64)
        episode["rewards"] = np.array(episode["rewards"], dtype=np.float32)
        
        return episode
    
    def normalize_returns(self, episodes: List[Dict]) -> np.ndarray:
        returns = np.array([e["return"] for e in episodes])
        
        if self.args.normalization_type == "zscore":
            # Z-score normalization
            mean = returns.mean()
            std = returns.std() + 1e-8
            normalized = (returns - mean) / std
            
            # Update running statistics
            self.return_stats["mean"] = mean
            self.return_stats["std"] = std
            
        elif self.args.normalization_type == "rank":
            # Rank-based normalization (0 to 1)
            ranks = np.argsort(np.argsort(returns))
            normalized = ranks / (len(ranks) - 1) if len(ranks) > 1 else ranks
            normalized = 2 * normalized - 1  # Scale to [-1, 1]
            
        elif self.args.normalization_type == "minmax":
            # Min-max normalization
            min_r, max_r = returns.min(), returns.max()
            if max_r - min_r > 0:
                normalized = 2 * (returns - min_r) / (max_r - min_r) - 1
            else:
                normalized = np.zeros_like(returns)
            
            self.return_stats["min"] = min_r
            self.return_stats["max"] = max_r
            
        elif self.args.normalization_type == "adaptive":
            # Adaptive normalization based on return distribution
            percentile_95 = np.percentile(returns, 95)
            percentile_5 = np.percentile(returns, 5)
            range_95 = percentile_95 - percentile_5 + 1e-8
            
            # Clip and normalize to handle outliers
            clipped = np.clip(returns, percentile_5, percentile_95)
            normalized = 2 * (clipped - percentile_5) / range_95 - 1
            
        elif self.args.normalization_type == "percentage":
            # Percentage-based normalization relative to maximum possible
            # This amplifies differences when close to maximum
            max_return = self.args.max_episode_return
            
            # Calculate percentage of maximum achieved
            percentages = returns / max_return
            
            # For differences, use log scale to amplify small differences near maximum
            # log(0.96) vs log(0.98) is more different than 0.96 vs 0.98
            log_percentages = np.log(np.clip(percentages, 0.01, 1.0))
            
            # Normalize to roughly [-1, 1] range
            # log(0.01) ≈ -4.6, log(1.0) = 0
            normalized = (log_percentages - np.mean(log_percentages)) / (np.std(log_percentages) + 1e-8)
            
        elif self.args.normalization_type == "advantage":
            # Advantage-style normalization (returns - baseline)
            # Uses exponential moving average as baseline
            mean = returns.mean()
            
            # Calculate advantages
            advantages = returns - mean
            
            # Normalize advantages
            std = advantages.std() + 1e-8
            normalized = advantages / std
            
        else:
            # No normalization
            normalized = returns
        
        return normalized
    
    def compute_preference_weights(self, good_episodes: List[Dict], bad_episodes: List[Dict]) -> tuple:
        if not self.args.normalize_returns:
            return None, None
        
        all_episodes = good_episodes + bad_episodes
        all_returns = np.array([e["return"] for e in all_episodes])
        
        # Apply temperature-scaled softmax to create smoother preferences
        if self.args.temperature_scaling != 1.0:
            # Softmax-based weighting
            scaled_returns = all_returns / self.args.temperature_scaling
            weights = np.exp(scaled_returns - np.max(scaled_returns))
            weights = weights / weights.sum()
            
            n_good = len(good_episodes)
            good_weights = weights[:n_good]
            bad_weights = weights[n_good:]
            
            return good_weights, bad_weights
        
        return None, None
    
    def dpo_step(self, good_episodes: List[Dict], bad_episodes: List[Dict]):
        losses = []
        
        # Store original returns for percentage calculations
        good_returns_orig = np.array([e["return"] for e in good_episodes])
        bad_returns_orig = np.array([e["return"] for e in bad_episodes])
        
        # Compute normalized scores if enabled
        if self.args.normalize_returns:
            all_episodes = good_episodes + bad_episodes
            normalized_scores = self.normalize_returns(all_episodes)
            
            # Split back into good/bad
            n_good = len(good_episodes)
            good_scores = normalized_scores[:n_good]
            bad_scores = normalized_scores[n_good:]
            
            # Compute adaptive beta based on score variance or proximity to max
            if self.args.adaptive_beta:
                if self.args.normalization_type == "percentage":
                    # Scale beta based on proximity to maximum
                    # As we get closer to max, increase beta to maintain learning
                    mean_return = np.mean([e["return"] for e in all_episodes])
                    proximity = mean_return / self.args.max_episode_return
                    # Exponentially increase beta as we approach maximum
                    adaptive_beta = self.args.beta * (1 + 2 * proximity ** 2)
                    adaptive_beta = np.clip(adaptive_beta, 0.01, 1.0)
                else:
                    score_std = normalized_scores.std() + 1e-8
                    # Scale beta inversely with variance (less variance = higher beta)
                    adaptive_beta = self.args.beta / score_std
                    adaptive_beta = np.clip(adaptive_beta, 0.01, 1.0)
            else:
                adaptive_beta = self.args.beta
        else:
            good_scores = good_returns_orig
            bad_scores = bad_returns_orig
            adaptive_beta = self.args.beta
        
        # Get preference weights
        good_weights, bad_weights = self.compute_preference_weights(good_episodes, bad_episodes)
        
        # Sample pairs for this gradient step
        n_pairs = min(self.args.batch_size, len(good_episodes), len(bad_episodes))
        for i in range(n_pairs):
            # Weighted sampling if using temperature scaling
            if good_weights is not None and bad_weights is not None:
                good_idx = np.random.choice(len(good_episodes), p=good_weights/good_weights.sum())
                bad_idx = np.random.choice(len(bad_episodes), p=bad_weights/bad_weights.sum())
            else:
                # Random sampling with direct index tracking
                good_idx = random.randint(0, len(good_episodes) - 1)
                bad_idx = random.randint(0, len(bad_episodes) - 1)
            
            good_ep = good_episodes[good_idx]
            bad_ep = bad_episodes[bad_idx]
            
            # Compute weight based on normalization type
            if self.args.normalize_returns:
                if self.args.normalization_type == "percentage":
                    # Use percentage difference directly as weight
                    good_return = good_returns_orig[good_idx]
                    bad_return = bad_returns_orig[bad_idx]
                    if bad_return > 0:
                        percentage_diff = (good_return - bad_return) / bad_return
                        weight = min(abs(percentage_diff), 2.0)  # Cap at 200% difference
                    else:
                        weight = 1.0
                else:
                    score_diff = good_scores[good_idx] - bad_scores[bad_idx]
                    weight = abs(score_diff)
            else:
                weight = 1.0
            
            # Convert to tensors
            good_obs = torch.tensor(good_ep["observations"], device=self.device)
            good_actions = torch.tensor(good_ep["actions"], device=self.device)
            bad_obs = torch.tensor(bad_ep["observations"], device=self.device)
            bad_actions = torch.tensor(bad_ep["actions"], device=self.device)
            
            # Compute log probabilities
            logp_good = self.policy.log_prob(good_obs, good_actions)
            logp_bad = self.policy.log_prob(bad_obs, bad_actions)
            
            if self.reference_policy is not None:
                with torch.no_grad():
                    ref_logp_good = self.reference_policy.log_prob(good_obs, good_actions)
                    ref_logp_bad = self.reference_policy.log_prob(bad_obs, bad_actions)
                
                # DPO with reference policy and adaptive beta
                logits = adaptive_beta * (
                    (logp_good - ref_logp_good) - 
                    (logp_bad - ref_logp_bad)
                )
            else:
                # Simple DPO without reference
                logits = adaptive_beta * (logp_good - logp_bad)
            
            # Bradley-Terry loss with optional weighting
            loss = -F.logsigmoid(logits) * weight
            losses.append(loss)
        
        # Optimize
        if losses:
            total_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            # Log adaptive beta if used
            if self.writer and self.args.adaptive_beta:
                self.writer.add_scalar("debug/adaptive_beta", adaptive_beta, self.global_step)
            
            return total_loss.item()
        return 0.0
    
    def train_step(self):
        self.iteration += 1
        
        # Collect new episodes
        print(f"\nIteration {self.iteration}: Collecting episodes...")
        new_episodes = []
        for _ in range(self.args.episodes_per_iteration):
            episode = self.collect_episode()
            new_episodes.append(episode)
            self.episodes.append(episode)
            
            if self.writer:
                self.writer.add_scalar("charts/episodic_return", episode["return"], self.global_step)
                self.writer.add_scalar("charts/episode_length", len(episode["actions"]), self.global_step)
        
        # Wait until we have enough episodes
        if len(self.episodes) < self.args.min_episodes_before_training:
            print(f"Collecting episodes... ({len(self.episodes)}/{self.args.min_episodes_before_training})")
            return
        
        # Get current return statistics
        current_returns = [e["return"] for e in self.episodes]
        return_std = np.std(current_returns)
        
        # Rank episodes by return
        sorted_episodes = sorted(self.episodes, key=lambda e: e["return"])
        n = len(sorted_episodes)
        
        # Adaptive percentile selection based on return variance
        if self.args.normalize_returns and return_std < 10:  # Tighten selection when returns are close
            percentile_gap = max(0.05, self.args.percentile_gap * (return_std / 50))
        else:
            percentile_gap = self.args.percentile_gap
        
        # Select good and bad episodes based on percentiles
        good_threshold = int((1 - percentile_gap/2) * n)
        bad_threshold = int(percentile_gap/2 * n)
        
        good_episodes = sorted_episodes[good_threshold:]
        bad_episodes = sorted_episodes[:bad_threshold]
        
        # Print statistics
        good_returns = [e["return"] for e in good_episodes]
        bad_returns = [e["return"] for e in bad_episodes]
        print(f"Good episodes (top {100*(1-good_threshold/n):.0f}%): "
              f"mean return = {np.mean(good_returns):.2f}")
        print(f"Bad episodes (bottom {100*bad_threshold/n:.0f}%): "
              f"mean return = {np.mean(bad_returns):.2f}")
        print(f"Buffer mean return: {np.mean(current_returns):.2f}, std: {return_std:.2f}")
        
        if self.args.normalize_returns:
            print(f"Normalization: {self.args.normalization_type}, "
                  f"Adaptive percentile gap: {percentile_gap:.3f}")
            if self.args.normalization_type == "percentage":
                mean_percentage = np.mean(current_returns) / self.args.max_episode_return * 100
                print(f"Mean performance: {mean_percentage:.1f}% of maximum")
        
        # Perform gradient steps
        total_loss = 0
        for _ in range(self.args.gradient_steps_per_iteration):
            loss = self.dpo_step(good_episodes, bad_episodes)
            total_loss += loss
        
        avg_loss = total_loss / self.args.gradient_steps_per_iteration
        print(f"Average DPO loss: {avg_loss:.4f}")
        
        # Log metrics
        if self.writer:
            self.writer.add_scalar("losses/dpo_loss", avg_loss, self.global_step)
            self.writer.add_scalar("debug/buffer_size", len(self.episodes), self.global_step)
            self.writer.add_scalar("debug/good_mean_return", np.mean(good_returns), self.global_step)
            self.writer.add_scalar("debug/bad_mean_return", np.mean(bad_returns), self.global_step)
            self.writer.add_scalar("debug/return_gap", 
                                  np.mean(good_returns) - np.mean(bad_returns), 
                                  self.global_step)
            self.writer.add_scalar("debug/return_std", return_std, self.global_step)
            self.writer.add_scalar("debug/percentile_gap", percentile_gap, self.global_step)
        
        # Update reference policy periodically
        if self.reference_policy and self.iteration % self.args.reference_update_freq == 0:
            self.reference_policy.load_state_dict(self.policy.state_dict())
            print("Updated reference policy")
    
    def train(self):
        start_time = time.time()
        
        while self.global_step < self.args.total_timesteps:
            self.train_step()
            
            if self.writer:
                sps = int(self.global_step / (time.time() - start_time))
                self.writer.add_scalar("charts/SPS", sps, self.global_step)
                print(f"SPS: {sps}")
        
        if self.writer:
            self.writer.close()


def main():
    args = tyro.cli(Args)
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = gym.make(args.env_id)
    
    # Set max return based on environment for percentage normalization
    if args.env_id == "CartPole-v1":
        args.max_episode_return = 500.0  # CartPole-v1 max is 500
    elif args.env_id == "CartPole-v0":
        args.max_episode_return = 200.0  # CartPole-v0 max is 200
    elif args.env_id == "Acrobot-v1":
        args.max_episode_return = -50.0 # Acrobot-v1 max is -50
    
    # Create and train agent
    agent = MinimalDPOAgent(env, args)
    agent.train()
    
    env.close()


if __name__ == "__main__":
    main()