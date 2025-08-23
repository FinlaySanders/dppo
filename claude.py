# PREFER: Preference-based Reinforcement Learning with Adaptive Mechanisms
import os
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
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = "prefer_adaptive"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "PREFER"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    
    # PREFER specific arguments
    buffer_size: int = 256
    """number of trajectories to store in buffer"""
    batch_size: int = 32
    """number of preference pairs per update"""
    update_frequency: int = 16
    """update policy every N trajectories collected"""
    update_epochs: int = 4
    """number of epochs per update"""
    
    # Adaptive components
    initial_beta: float = 10.0
    """initial temperature for DPO loss (will be adapted)"""
    adaptive_beta: bool = True
    """whether to adapt beta based on log prob scale"""
    gamma: float = 0.99
    """discount factor"""
    use_reference: bool = True
    """whether to use reference policy for KL regularization"""
    adaptive_ref_update: bool = True
    """whether to adapt reference update frequency based on KL divergence"""
    initial_ref_update_freq: int = 1000
    """initial reference update frequency in timesteps"""
    
    # Preference generation
    use_return_per_step: bool = False
    """whether to use return-per-step for preferences (better for varying lengths)"""
    adaptive_min_gap: bool = True
    """whether to adapt minimum return gap based on return variance"""
    min_gap_percentile: float = 0.2
    """minimum gap as percentile of return range"""
    
    # Regularization
    entropy_coef: float = 0.01
    """entropy regularization coefficient"""
    adaptive_entropy: bool = True
    """whether to adapt entropy coefficient based on policy entropy"""
    target_entropy: float = None
    """target entropy (if None, set based on action space)"""


def make_env(env_id, idx, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def forward(self, x):
        return self.net(x)
    
    def get_action(self, x):
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()
    
    def evaluate_actions(self, x, actions):
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        return probs.log_prob(actions), probs.entropy()


class TrajectoryBuffer:
    """Buffer storing complete trajectories with adaptive statistics"""
    
    def __init__(self, capacity: int):
        self.trajectories = deque(maxlen=capacity)
        self.return_stats = {"mean": 0, "std": 1, "min": 0, "max": 1}
        
    def add(self, trajectory: Dict):
        self.trajectories.append(trajectory)
        self._update_stats()
    
    def _update_stats(self):
        """Update return statistics for adaptive thresholds"""
        if len(self.trajectories) > 10:
            returns = [t["return"] for t in self.trajectories]
            self.return_stats = {
                "mean": np.mean(returns),
                "std": np.std(returns) + 1e-8,
                "min": np.min(returns),
                "max": np.max(returns)
            }
    
    def get_adaptive_min_gap(self, percentile: float = 0.2) -> float:
        """Get adaptive minimum gap based on return distribution"""
        return_range = self.return_stats["max"] - self.return_stats["min"]
        return max(0.1, return_range * percentile)
    
    def sample_pairs(self, n_pairs: int, min_gap: float = 0.0, 
                    use_return_per_step: bool = False) -> List[Tuple[Dict, Dict]]:
        """Sample pairs with adaptive criteria"""
        if len(self.trajectories) < 2:
            return []
        
        pairs = []
        attempts = 0
        max_attempts = n_pairs * 20
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            idx1, idx2 = random.sample(range(len(self.trajectories)), 2)
            traj1 = self.trajectories[idx1]
            traj2 = self.trajectories[idx2]
            
            # Choose comparison metric
            if use_return_per_step:
                metric1 = traj1.get("return_per_step", traj1["return"])
                metric2 = traj2.get("return_per_step", traj2["return"])
            else:
                metric1 = traj1["return"]
                metric2 = traj2["return"]
            
            # Check if gap is sufficient
            if abs(metric1 - metric2) >= min_gap:
                pairs.append((traj1, traj2))
        
        return pairs
    
    def get_length_return_correlation(self) -> float:
        """Compute correlation between trajectory length and return"""
        if len(self.trajectories) < 10:
            return 0.0
        
        lengths = [t["length"] for t in self.trajectories]
        returns = [t["return"] for t in self.trajectories]
        
        if len(set(lengths)) > 1:
            return np.corrcoef(lengths, returns)[0, 1]
        return 0.0
    
    def __len__(self):
        return len(self.trajectories)


class AdaptivePREFER:
    def __init__(self, envs, args: Args):
        self.envs = envs
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        
        # Networks
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        n_actions = envs.single_action_space.n
        
        self.policy = PolicyNetwork(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.learning_rate, eps=1e-5)
        
        # Reference policy
        if args.use_reference:
            self.reference_policy = PolicyNetwork(obs_dim, n_actions).to(self.device)
            self.reference_policy.load_state_dict(self.policy.state_dict())
        else:
            self.reference_policy = None
        
        # Trajectory buffer
        self.buffer = TrajectoryBuffer(args.buffer_size)
        
        # Adaptive parameters
        self.beta = args.initial_beta
        self.log_prob_ema = None
        self.entropy_ema = None
        self.kl_divergence_ema = None
        self.ref_update_freq = args.initial_ref_update_freq
        self.entropy_coef = args.entropy_coef
        
        # Target entropy (for adaptive entropy regularization)
        if args.target_entropy is None:
            # Heuristic: 50% of maximum entropy
            self.target_entropy = 0.5 * np.log(n_actions)
        else:
            self.target_entropy = args.target_entropy
        
        # Tracking
        self.global_step = 0
        self.trajectories_collected = 0
        self.updates_performed = 0
        self.last_ref_update = 0
        
    def collect_trajectories(self, n_trajectories: int) -> List[Dict]:
        """Collect complete trajectories with per-step metrics"""
        trajectories = []
        
        obs, _ = self.envs.reset()
        obs = torch.Tensor(obs).to(self.device)
        
        env_trajectories = [{"obs": [], "actions": [], "rewards": [], "done": False} 
                           for _ in range(self.args.num_envs)]
        
        while len(trajectories) < n_trajectories:
            with torch.no_grad():
                actions, log_probs, entropies = self.policy.get_action(obs)
            
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminations, truncations)
            
            for i in range(self.args.num_envs):
                if not env_trajectories[i]["done"]:
                    env_trajectories[i]["obs"].append(obs[i].cpu().numpy())
                    env_trajectories[i]["actions"].append(actions[i].cpu().numpy())
                    env_trajectories[i]["rewards"].append(rewards[i])
                    
                    if dones[i]:
                        traj = env_trajectories[i]
                        
                        # Compute returns (both total and per-step)
                        returns = 0
                        for r in reversed(traj["rewards"]):
                            returns = r + self.args.gamma * returns
                        
                        trajectory = {
                            "obs": np.array(traj["obs"], dtype=np.float32),
                            "actions": np.array(traj["actions"], dtype=np.int64),
                            "rewards": np.array(traj["rewards"], dtype=np.float32),
                            "return": returns,
                            "return_per_step": returns / len(traj["rewards"]),
                            "length": len(traj["rewards"])
                        }
                        
                        trajectories.append(trajectory)
                        self.global_step += trajectory["length"]
                        
                        env_trajectories[i] = {"obs": [], "actions": [], "rewards": [], "done": False}
                        
                        if len(trajectories) >= n_trajectories:
                            break
            
            obs = torch.Tensor(next_obs).to(self.device)
        
        return trajectories[:n_trajectories]
    
    def compute_adaptive_beta(self, log_p_w: torch.Tensor, log_p_l: torch.Tensor) -> float:
        """Adapt beta based on log probability scale"""
        if not self.args.adaptive_beta:
            return self.beta
        
        # Track typical log prob magnitude
        avg_log_prob = (log_p_w.item() + log_p_l.item()) / 2
        
        if self.log_prob_ema is None:
            self.log_prob_ema = avg_log_prob
        else:
            self.log_prob_ema = 0.95 * self.log_prob_ema + 0.05 * avg_log_prob
        
        # Scale beta to keep the difference in reasonable range
        # Target: beta * log_prob_diff should be in [-5, 5]
        typical_magnitude = max(abs(self.log_prob_ema), 0.1)
        adaptive_beta = self.args.initial_beta * (1.0 / typical_magnitude)
        
        return adaptive_beta
    
    def compute_adaptive_entropy_coef(self, entropy: float) -> float:
        """Adapt entropy coefficient to maintain target entropy"""
        if not self.args.adaptive_entropy:
            return self.entropy_coef
        
        # Track entropy
        if self.entropy_ema is None:
            self.entropy_ema = entropy
        else:
            self.entropy_ema = 0.95 * self.entropy_ema + 0.05 * entropy
        
        # Increase entropy coef if entropy too low, decrease if too high
        entropy_error = self.target_entropy - self.entropy_ema
        self.entropy_coef = self.entropy_coef * (1 + 0.01 * entropy_error)
        self.entropy_coef = np.clip(self.entropy_coef, 0.0001, 0.1)
        
        return self.entropy_coef
    
    def compute_preference_loss(self, pairs: List[Tuple[Dict, Dict]]) -> Tuple[torch.Tensor, Dict]:
        """Compute DPO loss with adaptive components"""
        if not pairs:
            return torch.tensor(0.0).to(self.device), {}
        
        losses = []
        entropy_bonuses = []
        kl_divergences = []
        
        for traj_w, traj_l in pairs:
            # Determine winner/loser based on chosen metric
            if self.args.use_return_per_step:
                metric_w = traj_w.get("return_per_step", traj_w["return"])
                metric_l = traj_l.get("return_per_step", traj_l["return"])
            else:
                metric_w = traj_w["return"]
                metric_l = traj_l["return"]
            
            if metric_w < metric_l:
                traj_w, traj_l = traj_l, traj_w
            
            # Convert to tensors
            obs_w = torch.FloatTensor(traj_w["obs"]).to(self.device)
            actions_w = torch.LongTensor(traj_w["actions"]).to(self.device)
            obs_l = torch.FloatTensor(traj_l["obs"]).to(self.device)
            actions_l = torch.LongTensor(traj_l["actions"]).to(self.device)
            
            # Compute log probabilities
            log_probs_w, entropy_w = self.policy.evaluate_actions(obs_w, actions_w)
            log_probs_l, entropy_l = self.policy.evaluate_actions(obs_l, actions_l)
            
            # Use mean to avoid length bias
            log_p_w = log_probs_w.mean()
            log_p_l = log_probs_l.mean()
            
            # Compute reference log probs if using reference
            if self.reference_policy is not None:
                with torch.no_grad():
                    ref_log_probs_w, _ = self.reference_policy.evaluate_actions(obs_w, actions_w)
                    ref_log_probs_l, _ = self.reference_policy.evaluate_actions(obs_l, actions_l)
                    ref_log_p_w = ref_log_probs_w.mean()
                    ref_log_p_l = ref_log_probs_l.mean()
                
                # Track KL divergence for adaptive reference updates
                kl_w = (log_p_w - ref_log_p_w).item()
                kl_l = (log_p_l - ref_log_p_l).item()
                kl_divergences.append((kl_w + kl_l) / 2)
                
                # DPO loss with KL regularization
                log_ratio_w = log_p_w - ref_log_p_w
                log_ratio_l = log_p_l - ref_log_p_l
            else:
                log_ratio_w = log_p_w
                log_ratio_l = log_p_l
            
            # Get adaptive beta
            adaptive_beta = self.compute_adaptive_beta(log_p_w, log_p_l)
            
            # DPO loss
            loss = -F.logsigmoid(adaptive_beta * (log_ratio_w - log_ratio_l))
            losses.append(loss)
            
            # Track entropy
            avg_entropy = (entropy_w.mean() + entropy_l.mean()) / 2
            entropy_bonuses.append(avg_entropy)
        
        # Combine losses
        total_loss = torch.stack(losses).mean()
        
        # Adaptive entropy regularization
        if self.args.adaptive_entropy and entropy_bonuses:
            avg_entropy = torch.stack(entropy_bonuses).mean()
            adaptive_entropy_coef = self.compute_adaptive_entropy_coef(avg_entropy.item())
            total_loss = total_loss - adaptive_entropy_coef * avg_entropy
        elif self.entropy_coef > 0 and entropy_bonuses:
            avg_entropy = torch.stack(entropy_bonuses).mean()
            total_loss = total_loss - self.entropy_coef * avg_entropy
        else:
            avg_entropy = torch.tensor(0.0)
        
        # Update KL tracking for adaptive reference updates
        if kl_divergences:
            avg_kl = np.mean(kl_divergences)
            if self.kl_divergence_ema is None:
                self.kl_divergence_ema = avg_kl
            else:
                self.kl_divergence_ema = 0.9 * self.kl_divergence_ema + 0.1 * avg_kl
        
        metrics = {
            "preference_loss": total_loss.item(),
            "entropy": avg_entropy.item(),
            "adaptive_beta": adaptive_beta,
            "entropy_coef": self.entropy_coef,
            "n_pairs": len(pairs),
        }
        
        if kl_divergences:
            metrics["kl_divergence"] = np.mean(kl_divergences)
        
        return total_loss, metrics
    
    def update(self):
        """Perform policy update with adaptive components"""
        if len(self.buffer) < 10:
            return None
        
        # Get adaptive minimum gap
        if self.args.adaptive_min_gap:
            min_gap = self.buffer.get_adaptive_min_gap(self.args.min_gap_percentile)
        else:
            min_gap = 0.1
        
        all_metrics = []
        
        for epoch in range(self.args.update_epochs):
            # Sample pairs with adaptive criteria
            pairs = self.buffer.sample_pairs(
                self.args.batch_size,
                min_gap=min_gap,
                use_return_per_step=self.args.use_return_per_step
            )
            
            if not pairs:
                continue
            
            # Compute and apply gradients
            loss, metrics = self.compute_preference_loss(pairs)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            metrics["min_gap"] = min_gap
            all_metrics.append(metrics)
        
        self.updates_performed += 1
        
        # Check if we should update reference (adaptive frequency)
        if self.args.adaptive_ref_update and self.reference_policy is not None:
            self.maybe_update_reference_adaptive()
        
        # Average metrics
        if all_metrics:
            avg_metrics = {
                key: np.mean([m[key] for m in all_metrics])
                for key in all_metrics[0].keys()
            }
            
            # Add diagnostics
            avg_metrics["length_return_correlation"] = self.buffer.get_length_return_correlation()
            avg_metrics["return_mean"] = self.buffer.return_stats["mean"]
            avg_metrics["return_std"] = self.buffer.return_stats["std"]
            
            return avg_metrics
        return None
    
    def maybe_update_reference_adaptive(self):
        """Adaptively update reference based on KL divergence"""
        if self.kl_divergence_ema is None:
            return
        
        # Update more frequently if KL is high, less if low
        if self.kl_divergence_ema > 2.0:  # High KL - policy changing fast
            self.ref_update_freq = max(500, self.ref_update_freq * 0.9)
        elif self.kl_divergence_ema < 0.5:  # Low KL - policy stable
            self.ref_update_freq = min(5000, self.ref_update_freq * 1.1)
        
        # Check if it's time to update
        if self.global_step - self.last_ref_update > self.ref_update_freq:
            self.update_reference()
            self.last_ref_update = self.global_step
    
    def update_reference(self):
        """Update reference policy"""
        if self.reference_policy is not None:
            self.reference_policy.load_state_dict(self.policy.state_dict())


def main():
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\\n|-|-|\\n%s" % ("\\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, run_name) for i in range(args.num_envs)]
    )
    
    # Create agent
    agent = AdaptivePREFER(envs, args)
    
    # Training loop
    start_time = time.time()
    
    while agent.global_step < args.total_timesteps:
        # Collect trajectories
        trajectories = agent.collect_trajectories(args.update_frequency)
        
        # Add to buffer and track stats
        episode_returns = []
        for traj in trajectories:
            agent.buffer.add(traj)
            episode_returns.append(traj["return"])
            agent.trajectories_collected += 1
            
            # Log episode stats
            writer.add_scalar("charts/episodic_return", traj["return"], agent.global_step)
            writer.add_scalar("charts/episodic_length", traj["length"], agent.global_step)
            writer.add_scalar("charts/return_per_step", traj["return_per_step"], agent.global_step)
        
        # Update policy
        metrics = agent.update()
        
        if metrics:
            for key, value in metrics.items():
                writer.add_scalar(f"losses/{key}", value, agent.global_step)
            
            # Log adaptive parameters
            writer.add_scalar("adaptive/beta", agent.beta, agent.global_step)
            writer.add_scalar("adaptive/entropy_coef", agent.entropy_coef, agent.global_step)
            writer.add_scalar("adaptive/ref_update_freq", agent.ref_update_freq, agent.global_step)
            
            # Log important diagnostics
            if abs(metrics["length_return_correlation"]) > 0.5:
                print(f"WARNING: High length-return correlation: {metrics['length_return_correlation']:.3f}")
        
        # Non-adaptive reference update (if not using adaptive)
        if args.use_reference and not args.adaptive_ref_update:
            if agent.global_step % args.initial_ref_update_freq == 0:
                agent.update_reference()
        
        # Logging
        if agent.trajectories_collected % 100 == 0:
            sps = int(agent.global_step / (time.time() - start_time))
            avg_return = np.mean(episode_returns) if episode_returns else 0
            
            print(f"Steps: {agent.global_step}, "
                  f"Trajectories: {agent.trajectories_collected}, "
                  f"Avg Return: {avg_return:.2f}, "
                  f"Buffer Size: {len(agent.buffer)}, "
                  f"Beta: {agent.beta:.3f}, "
                  f"SPS: {sps}")
            
            writer.add_scalar("charts/SPS", sps, agent.global_step)
            writer.add_scalar("charts/avg_episodic_return", avg_return, agent.global_step)
    
    envs.close()
    writer.close()
    
    print(f"\\nTraining complete!")
    print(f"Total timesteps: {agent.global_step}")
    print(f"Total trajectories: {agent.trajectories_collected}")
    
    # Final evaluation
    print("\\nEvaluating final policy...")
    eval_env = gym.make(args.env_id)
    eval_returns = []
    
    for _ in range(20):
        obs, _ = eval_env.reset()
        done = False
        episode_return = 0
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action, _, _ = agent.policy.get_action(obs_tensor)
            obs, reward, terminated, truncated, _ = eval_env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            episode_return += reward
        
        eval_returns.append(episode_return)
    
    print(f"Final evaluation over 20 episodes: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")


if __name__ == "__main__":
    main()