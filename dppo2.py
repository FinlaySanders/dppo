import random
import time
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple

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
    exp_name: str = "oppo"
    seed: int = 1
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "oppo"
    wandb_entity: str = None
    
    # Core hyperparameters
    env_id: str = "CartPole-v1"
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    
    # OPPO hyperparameters
    beta: float = 0.1  # Fixed beta for DPO loss
    buffer_size: int = 32  # Size of trajectory buffer
    min_buffer_size: int = 10  # Minimum episodes before training starts
    pairs_per_update: int = 128  # Number of preference pairs per update
    update_frequency: int = 1  # Update policy every N episodes
    gamma: float = 0.99  # Discount factor
    
    # Reference policy
    tau: float = 0.005  # Soft update parameter for reference policy


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
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        logits = self.forward(obs)
        if deterministic:
            return torch.argmax(logits).item()
        dist = Categorical(logits=logits)
        return dist.sample().item()
    
    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions).sum()  # Sum over trajectory


class TrajectoryBuffer:
    """Buffer to store trajectories with their returns."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, episode: Dict):
        """Add an episode to the buffer."""
        self.buffer.append(episode)
    
    def sample_pairs(self, n_pairs: int) -> List[Tuple[Dict, Dict]]:
        """Sample pairs of trajectories for preference learning."""
        pairs = []
        
        if len(self.buffer) < 2:
            return pairs
        
        for _ in range(n_pairs):
            # Sample two different trajectories
            indices = random.sample(range(len(self.buffer)), 2)
            traj1 = self.buffer[indices[0]]
            traj2 = self.buffer[indices[1]]
            
            # Only create pair if returns are different (avoid ties)
            if abs(traj1["return"] - traj2["return"]) > 1e-6:
                pairs.append((traj1, traj2))
        
        return pairs
    
    def __len__(self):
        return len(self.buffer)


class OPPO:
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
        
        # Trajectory buffer
        self.buffer = TrajectoryBuffer(args.buffer_size)
        
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name=f"oppo3e-3lunar_",
            )
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
        
        self.global_step = 0
        self.episode_count = 0
    
    def collect_episode(self) -> Dict:
        """Collect a single trajectory using current policy."""
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
            self.global_step += 1
        
        # Calculate discounted return
        returns = 0
        discounted_return = 0
        for r in reversed(episode["rewards"]):
            returns = r + self.args.gamma * returns
            discounted_return = r + discounted_return
        
        episode["return"] = discounted_return  # Can use discounted_return or undiscounted
        episode["observations"] = np.array(episode["observations"], dtype=np.float32)
        episode["actions"] = np.array(episode["actions"], dtype=np.int64)
        episode["rewards"] = np.array(episode["rewards"], dtype=np.float32)
        
        self.episode_count += 1
        
        if self.args.track:
            wandb.log({
                "charts/episodic_return": episode["return"],
                "charts/episode_length": len(episode["actions"]),
                "global_step": self.global_step
            }, step=self.global_step)
        
        return episode
    
    def compute_preference_loss(self, pairs: List[Tuple[Dict, Dict]]) -> torch.Tensor:
        """Compute DPO-style preference loss."""
        if not pairs:
            return torch.tensor(0.0, device=self.device)
        
        losses = []
        
        for traj1, traj2 in pairs:
            # Determine winner and loser based on returns
            if traj1["return"] > traj2["return"]:
                winner, loser = traj1, traj2
            else:
                winner, loser = traj2, traj1
            
            # Convert to tensors
            winner_obs = torch.tensor(winner["observations"], device=self.device)
            winner_actions = torch.tensor(winner["actions"], device=self.device)
            loser_obs = torch.tensor(loser["observations"], device=self.device)
            loser_actions = torch.tensor(loser["actions"], device=self.device)
            
            # Compute log probabilities under current policy
            log_p_winner = self.policy.log_prob(winner_obs, winner_actions)
            log_p_loser = self.policy.log_prob(loser_obs, loser_actions)
            
            # Compute log probabilities under reference policy (for KL regularization)
            with torch.no_grad():
                ref_log_p_winner = self.reference_policy.log_prob(winner_obs, winner_actions)
                ref_log_p_loser = self.reference_policy.log_prob(loser_obs, loser_actions)
            
            # DPO loss with KL regularization from reference policy
            log_ratio_winner = log_p_winner - ref_log_p_winner
            log_ratio_loser = log_p_loser - ref_log_p_loser
            
            # Standard DPO loss: -log sigmoid(beta * (log_ratio_winner - log_ratio_loser))
            loss = -F.logsigmoid(self.args.beta * (log_ratio_winner - log_ratio_loser))
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def update_policy(self):
        """Perform a single policy update using preferences from buffer."""
        if len(self.buffer) < self.args.min_buffer_size:
            return None
        
        # Sample preference pairs from the entire buffer
        pairs = self.buffer.sample_pairs(self.args.pairs_per_update)
        
        if not pairs:
            return None
        
        # Compute and apply gradients
        self.optimizer.zero_grad()
        loss = self.compute_preference_loss(pairs)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        if self.args.track:
            wandb.log({
                "loss/preference": loss.item(),
                "debug/buffer_size": len(self.buffer),
                "debug/n_pairs": len(pairs),
            }, step=self.global_step)
        
        return loss.item()
    
    def soft_update_reference_policy(self):
        """Soft update of reference policy using exponential moving average."""
        for target_param, param in zip(self.reference_policy.parameters(), 
                                       self.policy.parameters()):
            target_param.data.copy_(
                self.args.tau * param.data + (1.0 - self.args.tau) * target_param.data
            )
    
    def train(self):
        start_time = time.time()
        losses = []
        
        while self.global_step < self.args.total_timesteps:
            # Collect trajectory
            episode = self.collect_episode()
            self.buffer.push(episode)
            
            # Update policy every N episodes
            if self.episode_count % self.args.update_frequency == 0:
                loss = self.update_policy()
                if loss is not None:
                    losses.append(loss)
                    
                # Soft update reference policy
                self.soft_update_reference_policy()
            
            # Logging
            if self.args.track and self.episode_count % 10 == 0:
                sps = int(self.global_step / (time.time() - start_time))
                avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else (np.mean(losses) if losses else 0)
                wandb.log({
                    "charts/SPS": sps,
                    "loss/avg_preference": avg_loss,
                }, step=self.global_step)
            
            # Print progress
            if self.episode_count % 100 == 0:
                recent_returns = [ep["return"] for ep in list(self.buffer.buffer)[-100:]]
                avg_return = np.mean(recent_returns) if recent_returns else 0
                print(f"Episode {self.episode_count}, "
                      f"Avg Return: {avg_return:.2f}, "
                      f"Buffer Size: {len(self.buffer)}, "
                      f"Global Step: {self.global_step}")
    
    def evaluate(self, n_episodes: int = 10) -> float:
        """Evaluate current policy."""
        total_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
                action = self.policy.get_action(obs_tensor, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)


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
    agent = OPPO(env, args)
    agent.train()
    
    # Final evaluation
    print("\nEvaluating final policy...")
    avg_reward = agent.evaluate(n_episodes=20)
    print(f"Final average reward: {avg_reward:.2f}")
    
    env.close()
    
    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()