import random
import time
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional

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
    exp_name: str = "oppo_segments"
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
    segment_length: int = 64  # Fixed size of trajectory segments
    buffer_size: int = 32  # Number of segments to keep in buffer
    min_buffer_size: int = 10  # Minimum segments before training starts
    pairs_per_update: int = 128  # Number of preference pairs per update
    update_frequency: int = 2  # Update policy every N segments
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


class Segment:
    """A fixed-size segment of experience that can cross episode boundaries."""
    
    def __init__(self, segment_length: int, gamma: float):
        self.segment_length = segment_length
        self.gamma = gamma
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.discounted_return = 0.0
        self.undiscounted_return = 0.0
    
    def add_step(self, obs: np.ndarray, action: int, reward: float, done: bool):
        """Add a single step to the segment."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def is_full(self) -> bool:
        """Check if segment has reached target length."""
        return len(self.actions) >= self.segment_length
    
    def finalize(self, bootstrap_value: Optional[float] = None):
        """Calculate returns for the segment."""
        # Convert lists to arrays
        self.observations = np.array(self.observations, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.int64)
        self.rewards = np.array(self.rewards, dtype=np.float32)
        self.dones = np.array(self.dones, dtype=bool)
        
        # Calculate returns
        self.undiscounted_return = np.sum(self.rewards)
        
        # Calculate discounted return with bootstrapping if segment doesn't end with done
        returns = bootstrap_value if bootstrap_value is not None and not self.dones[-1] else 0
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                returns = self.rewards[i]  # Reset at episode boundary
            else:
                returns = self.rewards[i] + self.gamma * returns
        
        self.discounted_return = returns
    
    def to_dict(self) -> Dict:
        """Convert segment to dictionary format."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "return": self.discounted_return,
            "undiscounted_return": self.undiscounted_return
        }


class SegmentBuffer:
    """Buffer to store fixed-size segments with their returns."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, segment: Dict):
        """Add a segment to the buffer."""
        self.buffer.append(segment)
    
    def sample_pairs(self, n_pairs: int) -> List[Tuple[Dict, Dict]]:
        """Sample pairs of segments for preference learning."""
        pairs = []
        
        if len(self.buffer) < 2:
            return pairs
        
        for _ in range(n_pairs):
            # Sample two different segments
            indices = random.sample(range(len(self.buffer)), 2)
            seg1 = self.buffer[indices[0]]
            seg2 = self.buffer[indices[1]]
            
            # Only create pair if returns are different (avoid ties)
            if abs(seg1["return"] - seg2["return"]) > 1e-6:
                pairs.append((seg1, seg2))
        
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
        
        # Segment buffer
        self.buffer = SegmentBuffer(args.buffer_size)
        
        # Current segment being collected
        self.current_segment = Segment(args.segment_length, args.gamma)
        
        # Episode tracking
        self.current_obs = None
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_count = 0
        
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name=f"oppo_segments_{args.env_id}_{args.seed}",
            )
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
        
        self.global_step = 0
        self.segment_count = 0
    
    def reset_episode(self):
        """Reset for a new episode."""
        self.current_obs, _ = self.env.reset()
        
        # Log episode metrics if we just finished one
        if self.episode_length > 0:
            self.episode_count += 1
            if self.args.track:
                wandb.log({
                    "charts/episodic_return": self.episode_reward,
                    "charts/episode_length": self.episode_length,
                    "global_step": self.global_step
                }, step=self.global_step)
        
        self.episode_reward = 0
        self.episode_length = 0
    
    def collect_segment(self) -> Dict:
        """Collect a fixed-size segment that may cross episode boundaries."""
        # Start new segment
        self.current_segment = Segment(self.args.segment_length, self.args.gamma)
        
        # If we don't have a current observation, start a new episode
        if self.current_obs is None:
            self.reset_episode()
        
        while not self.current_segment.is_full():
            # Get action from policy
            obs_tensor = torch.tensor(self.current_obs, device=self.device, dtype=torch.float32)
            action = self.policy.get_action(obs_tensor)
            
            # Take environment step
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Add to segment
            self.current_segment.add_step(self.current_obs, action, reward, done)
            
            # Update episode tracking
            self.episode_reward += reward
            self.episode_length += 1
            self.global_step += 1
            
            if done:
                # Episode ended, start new one
                self.reset_episode()
            else:
                # Continue with next observation
                self.current_obs = next_obs
        
        # Bootstrap value for incomplete segments
        bootstrap_value = None
        if not self.current_segment.dones[-1]:
            with torch.no_grad():
                obs_tensor = torch.tensor(self.current_obs, device=self.device, dtype=torch.float32)
                logits = self.policy(obs_tensor)
                # Use value of max action as bootstrap (Q-learning style)
                # Could also use expected value under policy
                bootstrap_value = 0  # Simple approach: assume 0 future value
        
        # Finalize segment
        self.current_segment.finalize(bootstrap_value)
        segment_dict = self.current_segment.to_dict()
        
        self.segment_count += 1
        
        if self.args.track:
            wandb.log({
                "charts/segment_return": segment_dict["return"],
                "charts/segment_undiscounted_return": segment_dict["undiscounted_return"],
                "debug/segment_count": self.segment_count,
                "global_step": self.global_step
            }, step=self.global_step)
        
        return segment_dict
    
    def compute_preference_loss(self, pairs: List[Tuple[Dict, Dict]]) -> torch.Tensor:
        """Compute DPO-style preference loss."""
        if not pairs:
            return torch.tensor(0.0, device=self.device)
        
        losses = []
        
        for seg1, seg2 in pairs:
            # Determine winner and loser based on returns
            if seg1["return"] > seg2["return"]:
                winner, loser = seg1, seg2
            else:
                winner, loser = seg2, seg1
            
            # Filter out steps after episode ends (optional, can keep them)
            # Here we keep all steps for simplicity
            
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
            # Collect segment
            segment = self.collect_segment()
            self.buffer.push(segment)
            
            # Update policy every N segments
            if self.segment_count % self.args.update_frequency == 0:
                loss = self.update_policy()
                if loss is not None:
                    losses.append(loss)
                    
                # Soft update reference policy
                self.soft_update_reference_policy()
            
            # Logging
            if self.args.track and self.segment_count % 10 == 0:
                sps = int(self.global_step / (time.time() - start_time))
                avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else (np.mean(losses) if losses else 0)
                wandb.log({
                    "charts/SPS": sps,
                    "loss/avg_preference": avg_loss,
                }, step=self.global_step)
            
            # Print progress
            if self.segment_count % 100 == 0:
                recent_returns = [seg["return"] for seg in list(self.buffer.buffer)[-10:]]
                avg_return = np.mean(recent_returns) if recent_returns else 0
                print(f"Segment {self.segment_count}, "
                      f"Episode {self.episode_count}, "
                      f"Avg Segment Return: {avg_return:.2f}, "
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