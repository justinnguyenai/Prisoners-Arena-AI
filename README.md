# Prisoners Arena

A comprehensive implementation of the Iterated Prisoner's Dilemma tournament featuring 20+ classic strategies plus a deep learning approach using reinforcement learning.

## Overview

This project implements Axelrod's famous Prisoner's Dilemma tournament, where different strategies compete against each other in repeated games. The tournament includes classic strategies like Tit-for-Tat, Always Cooperate, Always Defect, and many others, plus a modern Deep Learning strategy trained using PPO (Proximal Policy Optimization) reinforcement learning.

## Features

- **20+ Classic Strategies**: Including Tit-for-Tat, Generous Tit-for-Tat, Pavlov, Grim Trigger, Detective, and more
- **Deep Learning Strategy**: PPO-based neural network that learns to play against various opponents
- **Tournament System**: Complete round-robin tournament with detailed statistics
- **Customizable Parameters**: Adjustable match lengths, payoff matrices, and tournament settings

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Deep Learning Model** (optional):
   ```bash
   python train_dl_strategy.py
   ```
   This will train a PPO model for 1 million timesteps and save it as `prisoner_dilemma_ppo_model.zip`.

## Usage

### Run Tournament
```bash
python prisoners_arena.py
```

This will run a complete tournament where each strategy plays against every other strategy. The Deep Learning strategy will automatically be included if the model file exists.

### Train Deep Learning Strategy
```bash
python train_dl_strategy.py
```

Trains a new deep learning model using PPO reinforcement learning. The model learns by playing against all 20 classic strategies and optimizing its performance.

## Strategies Included

### Classic Strategies
- **Tit for Tat**: Cooperate first, then copy opponent's last move
- **Tit for Two Tats**: Cooperate unless opponent defected twice in a row
- **Generous Tit for Tat**: Like TFT but 10% chance to forgive defections
- **Pavlov**: Win-stay, lose-shift strategy
- **Always Defect**: Always defect
- **Always Cooperate**: Always cooperate
- **Grim Trigger**: Cooperate until opponent defects once, then always defect
- **Tester**: Defect first to test opponent's response
- **Joss**: TFT with 10% random defection
- **Random**: 50% chance of cooperation or defection
- **Grudger**: Punish defection with temporary retaliation
- **Suspicious Tit for Tat**: Defect first, then copy opponent
- **Alternator**: Alternate between cooperate and defect
- **Adaptive**: Learn opponent's cooperation rate and adapt
- **Detective**: Test with C,D,C,C pattern then adapt strategy
- **Prober**: Probe with D,C,C pattern and exploit if possible
- **Forgiving Tit for Tat**: TFT with occasional forgiveness
- **Spiteful**: Cooperate until betrayed, then play randomly with bias toward defection
- **Bully**: Defect until opponent fights back
- **Remorseful Prober**: Like Prober but shows remorse after defecting

### Deep Learning Strategy
- **Deep Learning**: PPO-trained neural network that observes move history and adapts its strategy

## Game Rules

- **Payoff Matrix**:
  - Both Cooperate: (3, 3)
  - Cooperate vs Defect: (0, 5)
  - Defect vs Cooperate: (5, 0)
  - Both Defect: (1, 1)

- **Match Structure**: Each match continues until random termination (average ~200 games per match)
- **Tournament**: Each pair of strategies plays 25 matches

## Technical Details

The Deep Learning strategy uses:
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network**: LSTM-based feature extractor with 128 hidden units
- **Observation Space**: Last 10 moves of both players + round progress (21 features)
- **Training**: 1 million timesteps against all classic strategies

## Results

The tournament displays:
- Average score per game for each strategy
- Total games played
- Detailed match statistics
- Strategy rankings

Typically, cooperative strategies with retaliation capabilities (like Tit-for-Tat variants) perform well, while the Deep Learning strategy adapts its behavior based on the opponent type.