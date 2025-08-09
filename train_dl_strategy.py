import numpy as np
import random
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import warnings
import sys
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import strategies from the main file
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prisoners_arena import (
    Strategy, TitForTat, TitForTwoTats, GenerousTitForTat, 
    Pavlov, AlwaysDefect, AlwaysCooperate, GrimTrigger, 
    Tester, Joss, RandomStrategy, Grudger, SuspiciousTitForTat,
    Alternator, Adaptive, Detective, Prober, ForgivingTitForTat,
    Spiteful, Bully, RemorsefulProber, COOPERATE, DEFECT, PAYOFF_MATRIX,
    MATCH_END_PROBABILITY
)


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """Custom LSTM feature extractor for PPO."""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        self.lstm = nn.LSTM(
            input_size=21,  # 10 own moves + 10 opponent moves + 1 round progress
            hidden_size=features_dim,
            batch_first=True
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape for LSTM: (batch, seq_len=1, features)
        x = observations.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        # Return last output
        return lstm_out[:, -1, :]


class PrisonersDilemmaEnv(gym.Env):
    """Custom Environment for Prisoner's Dilemma with opponent strategies."""
    
    def __init__(self):
        super().__init__()
        
        # Action space: 0 = Defect, 1 = Cooperate
        self.action_space = spaces.Discrete(2)
        
        # Observation space: 10 own moves + 10 opponent moves + round progress
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(21,), dtype=np.float32
        )
        
        # Initialize all 20 opponent strategies
        self.strategies = [
            TitForTat(), TitForTwoTats(), GenerousTitForTat(),
            Pavlov(), AlwaysDefect(), AlwaysCooperate(),
            GrimTrigger(), Tester(), Joss(), RandomStrategy(),
            Grudger(), SuspiciousTitForTat(), Alternator(),
            Adaptive(), Detective(), Prober(), ForgivingTitForTat(),
            Spiteful(), Bully(), RemorsefulProber()
        ]
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the environment for a new match."""
        super().reset(seed=seed)
        
        # Select random opponent
        self.opponent = random.choice(self.strategies)
        self.opponent.reset()
        
        # Reset histories
        self.own_history = []
        self.opponent_history = []
        self.round = 0
        self.total_reward = 0
        
        return self._get_observation(), {}
        
    def step(self, action):
        """Execute one step in the environment."""
        # Convert action to move
        my_move = COOPERATE if action == 1 else DEFECT
        
        # Get opponent's move
        opponent_move = self.opponent.play()
        
        # Calculate rewards
        my_reward, opponent_reward = PAYOFF_MATRIX[(my_move, opponent_move)]
        
        # Update histories
        self.own_history.append(my_move)
        self.opponent_history.append(opponent_move)
        self.opponent.update_history(opponent_move, my_move, opponent_reward)
        
        self.round += 1
        self.total_reward += my_reward
        
        # Check if match ends
        terminated = random.random() < MATCH_END_PROBABILITY
        truncated = False
        
        observation = self._get_observation()
        
        return observation, my_reward, terminated, truncated, {}
        
    def _get_observation(self):
        """Get current observation."""
        obs = np.zeros(21, dtype=np.float32)
        
        # Last 10 own moves (C=1, D=0) - chronological order
        history_len = len(self.own_history)
        for i in range(min(10, history_len)):
            # Get moves in chronological order (0 = oldest, 9 = newest of last 10)
            move_index = history_len - min(10, history_len) + i
            if self.own_history[move_index] == COOPERATE:
                obs[i] = 1.0
                
        # Last 10 opponent moves - chronological order  
        opp_history_len = len(self.opponent_history)
        for i in range(min(10, opp_history_len)):
            move_index = opp_history_len - min(10, opp_history_len) + i
            if self.opponent_history[move_index] == COOPERATE:
                obs[10+i] = 1.0
                
        # Round progress (normalized)
        obs[20] = min(self.round / 200.0, 1.0)
        
        return obs


class ProgressCallback(BaseCallback):
    """Callback to print training progress with tqdm bar."""
    
    def __init__(self, print_freq=5000, total_timesteps=200000):
        super().__init__()
        self.print_freq = print_freq
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.timesteps = 0
        self.pbar = None
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"Training Deep Learning Strategy for {self.total_timesteps:,} timesteps")
        print("-" * 60)
        self.pbar = tqdm(total=self.total_timesteps, 
                        desc="Training Progress", 
                        unit="step",
                        ncols=100)
        
    def _on_step(self) -> bool:
        self.timesteps += 1
        
        if self.pbar:
            self.pbar.update(1)
            
            if self.timesteps % self.print_freq == 0:
                elapsed_time = time.time() - self.start_time
                progress_pct = (self.timesteps / self.total_timesteps) * 100
                
                # Update progress bar description with stats
                elapsed_str = f"{int(elapsed_time/60)}m{int(elapsed_time%60):02d}s"
                self.pbar.set_description(f"Training Progress - Elapsed: {elapsed_str}")
                
        return True
        
    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
        total_time = time.time() - self.start_time
        print(f"\nTraining completed in {int(total_time/60)}m{int(total_time%60):02d}s")


def train_dl_strategy():
    """Train the deep learning strategy."""
    
    # Create environment 
    env = DummyVecEnv([lambda: PrisonersDilemmaEnv()])
    
    # Create PPO model with custom LSTM feature extractor
    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[],  # No additional layers after LSTM
        activation_fn=nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0
    )
    
    callback = ProgressCallback(print_freq=25_000, total_timesteps=1_000_000)
    model.learn(total_timesteps=1_000_000, callback=callback)
    
    # Save the trained model
    model.save("prisoner_dilemma_ppo_model")
    print("\nTraining complete! Model saved as 'prisoner_dilemma_ppo_model.zip'")
    
    # Test the trained model
    print("\nTesting trained model against each strategy (10 matches each):")
    print("-" * 60)
    
    env = PrisonersDilemmaEnv()  # Create testing environment
    
    for strategy in env.strategies:
        total_reward = 0
        total_games = 0
        
        for _ in range(10):
            # Reset environment state without changing opponent
            env.own_history = []
            env.opponent_history = []
            env.round = 0
            env.total_reward = 0
            
            # Set and reset the specific opponent we want to test against
            env.opponent = strategy
            env.opponent.reset()
            
            obs = env._get_observation()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                total_games += 1
                done = terminated or truncated
                
        avg_reward = total_reward / total_games
        print(f"vs {strategy.name:<20}: {avg_reward:.3f} points/game")
        
    return model


class DeepLearningStrategy(Strategy):
    """Deep Learning strategy using trained PPO model."""
    
    def __init__(self, model_path="prisoner_dilemma_ppo_model"):
        super().__init__("Deep Learning")
        self.model = PPO.load(model_path)
        self.round = 0
        
    def reset(self):
        """Reset strategy for a new match."""
        super().reset()
        self.round = 0
        
    def play(self) -> str:
        """Return the next move using the trained model."""
        # Prepare observation
        obs = np.zeros(21, dtype=np.float32)
        
        # Last 10 own moves - chronological order
        history_len = len(self.history)
        for i in range(min(10, history_len)):
            # Get moves in chronological order (0 = oldest, 9 = newest of last 10)
            move_index = history_len - min(10, history_len) + i
            if self.history[move_index] == COOPERATE:
                obs[i] = 1.0
                
        # Last 10 opponent moves - chronological order
        opp_history_len = len(self.opponent_history)
        for i in range(min(10, opp_history_len)):
            move_index = opp_history_len - min(10, opp_history_len) + i
            if self.opponent_history[move_index] == COOPERATE:
                obs[10+i] = 1.0
                
        # Round progress
        obs[20] = min(self.round / 200.0, 1.0)
        
        # Get action from model
        action, _ = self.model.predict(obs, deterministic=True)
        
        self.round += 1
        
        return COOPERATE if action == 1 else DEFECT
        
    def update_history(self, my_move: str, opponent_move: str, payoff: int):
        """Update move history."""
        super().update_history(my_move, opponent_move, payoff)


if __name__ == "__main__":
    # Train the model
    train_dl_strategy()
    
    # Create and test the strategy
    dl_strategy = DeepLearningStrategy()
    print(f"\nCreated {dl_strategy.name} strategy successfully!")