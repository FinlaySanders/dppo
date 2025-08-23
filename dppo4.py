# Sorted Buffer Percentile (SBP) Algorithm
import os
import random
import time
from dataclasses import dataclass
import bisect

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = "sbp"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    
    # SBP specific arguments
    initial_percentile: float = 0.2
    """initial percentile target to beat"""
    target_percentile: float = 0.9
    """maximum percentile target"""
    percentile_increment: float = 0.05
    """how much to increase percentile when successful"""
    success_threshold: float = 0.7
    """success rate needed to increase percentile"""
    buffer_size: int = 10000
    """maximum size of sorted buffer"""
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SortedBuffer:
    """Maintains a sorted buffer of trajectories by return"""
    def __init__(self, max_size=10000):
        self.trajectories = []  # List of (return, trajectory_data) tuples
        self.returns = []  # Just the returns for bisect
        self.max_size = max_size
    
    def add(self, trajectory_return, trajectory_data):
        """Insert trajectory maintaining sorted order"""
        idx = bisect.bisect_left(self.returns, trajectory_return)
        self.returns.insert(idx, trajectory_return)
        self.trajectories.insert(idx, (trajectory_return, trajectory_data))
        
        # Maintain max size by removing worst trajectories
        if len(self.trajectories) > self.max_size:
            self.trajectories.pop(0)
            self.returns.pop(0)
    
    def get_percentile(self, p):
        """Get return value at percentile p (0-1)"""
        if len(self.returns) == 0:
            return 0
        idx = min(int(p * len(self.returns)), len(self.returns) - 1)
        return self.returns[idx]
    
    def get_rank(self, trajectory_return):
        """Get percentile rank of a return value"""
        if len(self.returns) == 0:
            return 0.5
        idx = bisect.bisect_left(self.returns, trajectory_return)
        return idx / len(self.returns)
    
    def __len__(self):
        return len(self.trajectories)


class Agent(nn.Module):
    """Simple policy network without value function"""
    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # SBP specific setup
    sorted_buffer = SortedBuffer(max_size=args.buffer_size)
    current_percentile = args.initial_percentile
    recent_successes = []
    
    # Storage for rollouts
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Episode tracking
    episode_returns = np.zeros(args.num_envs)
    episode_lengths = np.zeros(args.num_envs)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect trajectories
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Get action
            with torch.no_grad():
                action, logprob, entropy = agent.get_action(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            # Execute action
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            # Track episode returns
            episode_returns += reward
            episode_lengths += 1

            if "episode" in infos:
                for i in range(args.num_envs):
                    if infos["episode"]["_r"][i]:
                        ep_return = infos["episode"]["r"][i]
                        ep_length = infos["episode"]["l"][i]
                        
                        # Add to sorted buffer
                        trajectory_data = {
                            'return': ep_return,
                            'length': ep_length,
                            'iteration': iteration
                        }
                        sorted_buffer.add(ep_return, trajectory_data)
                        
                        # Track success rate
                        target_return = sorted_buffer.get_percentile(current_percentile)
                        success = ep_return > target_return
                        recent_successes.append(success)
                        if len(recent_successes) > 20:  # Keep last 20 episodes
                            recent_successes.pop(0)
                        
                        # Logging
                        print(f"global_step={global_step}, return={ep_return:.2f}, target={target_return:.2f}, percentile={current_percentile:.2f}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        writer.add_scalar("charts/target_return", target_return, global_step)
                        writer.add_scalar("charts/current_percentile", current_percentile, global_step)
                        writer.add_scalar("charts/buffer_size", len(sorted_buffer), global_step)
                        
                        # Reset episode tracking for this env
                        episode_returns[i] = 0
                        episode_lengths[i] = 0

        # Calculate advantages based on percentile ranking
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_dones = dones.reshape(-1)
        
        # Compute trajectory returns for each segment
        trajectory_returns = []
        for env_idx in range(args.num_envs):
            env_rewards = rewards[:, env_idx]
            env_dones = dones[:, env_idx]
            
            # Simple cumulative return for each step
            for step in range(args.num_steps):
                # Calculate return from this step to episode end or batch end
                G = 0
                for t in range(step, args.num_steps):
                    G += env_rewards[t].item()
                    if env_dones[t]:
                        break
                trajectory_returns.append(G)
        
        trajectory_returns = torch.tensor(trajectory_returns, device=device)
        
        # Calculate advantages based on whether trajectory beats target percentile
        target_return = sorted_buffer.get_percentile(current_percentile)
        advantages = (trajectory_returns > target_return).float() - 0.5  # Center around 0
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std()
        
        # Policy gradient update
        for _ in range(4):  # Multiple epochs
            # Shuffle batch
            indices = torch.randperm(args.batch_size)
            
            for start in range(0, args.batch_size, args.batch_size // 4):  # Mini-batches
                end = min(start + args.batch_size // 4, args.batch_size)
                mb_indices = indices[start:end]
                
                # Get new log probs
                _, newlogprob, entropy = agent.get_action(b_obs[mb_indices], b_actions.long()[mb_indices])
                
                # Simple policy gradient loss
                ratio = (newlogprob - b_logprobs[mb_indices]).exp()
                pg_loss = -(advantages[mb_indices] * ratio).mean()
                
                # Add entropy bonus
                entropy_loss = entropy.mean()
                loss = pg_loss - 0.01 * entropy_loss
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        
        # Adjust percentile target based on success rate
        if len(recent_successes) >= 10:
            success_rate = np.mean(recent_successes)
            writer.add_scalar("charts/success_rate", success_rate, global_step)
            
            if success_rate > args.success_threshold:
                current_percentile = min(current_percentile + args.percentile_increment, args.target_percentile)
                print(f"Increasing target percentile to {current_percentile:.2f}")
        
        # Logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()