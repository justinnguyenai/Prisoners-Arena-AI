import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from collections import defaultdict
import os
import sys
from tqdm import tqdm

# Game constants
COOPERATE = 'C'
DEFECT = 'D'

# Payoff matrix
PAYOFF_MATRIX = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1)
}

# Tournament constants
MATCH_END_PROBABILITY = 1/200  # Probability of ending after each game
MATCHES_PER_PAIR = 25  # Number of matches between each pair of strategies


class Strategy(ABC):
    """Base class for all strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.history = []  # Own moves
        self.opponent_history = []  # Opponent's moves
        self.last_payoff = None
        
    def reset(self):
        """Reset strategy for a new match."""
        self.history = []
        self.opponent_history = []
        self.last_payoff = None
        
    def update_history(self, my_move: str, opponent_move: str, payoff: int):
        """Update move history and last payoff."""
        self.history.append(my_move)
        self.opponent_history.append(opponent_move)
        self.last_payoff = payoff
        
    @abstractmethod
    def play(self) -> str:
        """Return the next move: COOPERATE or DEFECT."""
        pass


class TitForTat(Strategy):
    """Cooperate first, then copy opponent's last move."""
    
    def __init__(self):
        super().__init__("Tit for Tat")
        
    def play(self) -> str:
        if not self.opponent_history:
            return COOPERATE
        return self.opponent_history[-1]


class TitForTwoTats(Strategy):
    """Cooperate unless opponent defected in both of the last two moves."""
    
    def __init__(self):
        super().__init__("Tit for Two Tats")
        
    def play(self) -> str:
        if len(self.opponent_history) < 2:
            return COOPERATE
        if self.opponent_history[-1] == DEFECT and self.opponent_history[-2] == DEFECT:
            return DEFECT
        return COOPERATE


class GenerousTitForTat(Strategy):
    """Like TFT but 10% chance to cooperate after opponent defects."""
    
    def __init__(self):
        super().__init__("Generous Tit for Tat")
        
    def play(self) -> str:
        if not self.opponent_history:
            return COOPERATE
        if self.opponent_history[-1] == DEFECT:
            if random.random() < 0.1:
                return COOPERATE
            return DEFECT
        return COOPERATE


class Pavlov(Strategy):
    """Repeat last move if got 3 or 5 points, switch if got 0 or 1."""
    
    def __init__(self):
        super().__init__("Pavlov")
        
    def play(self) -> str:
        if not self.history:
            return COOPERATE
        if self.last_payoff in [3, 5]:  # Win-stay
            return self.history[-1]
        else:  # Lose-shift
            return DEFECT if self.history[-1] == COOPERATE else COOPERATE


class AlwaysDefect(Strategy):
    """Always play D."""
    
    def __init__(self):
        super().__init__("Always Defect")
        
    def play(self) -> str:
        return DEFECT


class AlwaysCooperate(Strategy):
    """Always play C."""
    
    def __init__(self):
        super().__init__("Always Cooperate")
        
    def play(self) -> str:
        return COOPERATE


class GrimTrigger(Strategy):
    """Cooperate until opponent defects once, then always defect."""
    
    def __init__(self):
        super().__init__("Grim Trigger")
        self.triggered = False
        
    def reset(self):
        super().reset()
        self.triggered = False
        
    def play(self) -> str:
        if DEFECT in self.opponent_history:
            self.triggered = True
        return DEFECT if self.triggered else COOPERATE


class Tester(Strategy):
    """Defect first. If opponent defects back, play TFT. If not, alternate C,D."""
    
    def __init__(self):
        super().__init__("Tester")
        self.mode = None  # 'tft' or 'alternate'
        
    def reset(self):
        super().reset()
        self.mode = None
        
    def play(self) -> str:
        if not self.history:
            return DEFECT
        
        if len(self.history) == 1:
            # Determine mode based on opponent's response to initial defect
            self.mode = 'tft' if self.opponent_history[0] == DEFECT else 'alternate'
            
        if self.mode == 'tft':
            # Play Tit for Tat
            return self.opponent_history[-1]
        else:
            # Alternate C, D
            return COOPERATE if len(self.history) % 2 == 1 else DEFECT


class Joss(Strategy):
    """Play TFT but randomly defect 10% of the time."""
    
    def __init__(self):
        super().__init__("Joss")
        
    def play(self) -> str:
        if not self.opponent_history:
            tft_move = COOPERATE
        else:
            tft_move = self.opponent_history[-1]
            
        # 10% chance to defect regardless
        if random.random() < 0.1:
            return DEFECT
        return tft_move


class RandomStrategy(Strategy):
    """50% chance of C or D each round."""
    
    def __init__(self):
        super().__init__("Random")
        
    def play(self) -> str:
        return COOPERATE if random.random() < 0.5 else DEFECT


class Grudger(Strategy):
    """Cooperate until opponent defects, then defect for 3 rounds before forgiving."""
    
    def __init__(self):
        super().__init__("Grudger")
        self.grudge_counter = 0
        
    def reset(self):
        super().reset()
        self.grudge_counter = 0
        
    def play(self) -> str:
        if self.grudge_counter > 0:
            self.grudge_counter -= 1
            return DEFECT
            
        if self.opponent_history and self.opponent_history[-1] == DEFECT:
            self.grudge_counter = 2  # Defect for 3 total rounds (this one + 2 more)
            return DEFECT
            
        return COOPERATE


class SuspiciousTitForTat(Strategy):
    """Defect first, then copy opponent's last move."""
    
    def __init__(self):
        super().__init__("Suspicious Tit for Tat")
        
    def play(self) -> str:
        if not self.opponent_history:
            return DEFECT
        return self.opponent_history[-1]


class Alternator(Strategy):
    """Alternate between cooperate and defect, starting with cooperate."""
    
    def __init__(self):
        super().__init__("Alternator")
        
    def play(self) -> str:
        return COOPERATE if len(self.history) % 2 == 0 else DEFECT


class Adaptive(Strategy):
    """Learns opponent's cooperation rate and matches it with slight bias toward cooperation."""
    
    def __init__(self):
        super().__init__("Adaptive")
        
    def play(self) -> str:
        if len(self.opponent_history) < 6:
            return COOPERATE
            
        # Calculate opponent's cooperation rate over last 10 moves
        recent_moves = self.opponent_history[-10:]
        coop_rate = recent_moves.count(COOPERATE) / len(recent_moves)
        
        # Slightly bias toward cooperation (add 0.1 to cooperation probability)
        coop_prob = min(1.0, coop_rate + 0.1)
        
        return COOPERATE if random.random() < coop_prob else DEFECT


class Detective(Strategy):
    """Test with C,D,C,C pattern, then play based on whether opponent retaliated."""
    
    def __init__(self):
        super().__init__("Detective")
        self.test_pattern = [COOPERATE, DEFECT, COOPERATE, COOPERATE]
        self.mode = None  # 'always_defect' or 'tit_for_tat'
        
    def reset(self):
        super().reset()
        self.mode = None
        
    def play(self) -> str:
        # First 4 moves: follow test pattern
        if len(self.history) < 4:
            return self.test_pattern[len(self.history)]
            
        # After test pattern, determine mode
        if self.mode is None:
            # Check if opponent retaliated to the defection in move 2
            # opponent_history[1] is the response to our move 2 (DEFECT)
            opponent_retaliated = len(self.opponent_history) > 1 and self.opponent_history[1] == DEFECT
            self.mode = 'tit_for_tat' if opponent_retaliated else 'always_defect'
            
        # Play according to determined mode
        if self.mode == 'always_defect':
            return DEFECT
        else:
            return self.opponent_history[-1]


class Prober(Strategy):
    """Probe with D,C,C. If no retaliation to first D, defect every 3rd move."""
    
    def __init__(self):
        super().__init__("Prober")
        self.probe_pattern = [DEFECT, COOPERATE, COOPERATE]
        self.strategy_mode = None  # 'probe_exploit' or 'tit_for_tat'
        
    def reset(self):
        super().reset()
        self.strategy_mode = None
        
    def play(self) -> str:
        # First 3 moves: follow probe pattern
        if len(self.history) < 3:
            return self.probe_pattern[len(self.history)]
            
        # After probe, determine strategy
        if self.strategy_mode is None:
            # Check if opponent retaliated to the initial defection
            # opponent_history[0] is the response to our initial DEFECT
            retaliated = len(self.opponent_history) > 0 and self.opponent_history[0] == DEFECT
            self.strategy_mode = 'tit_for_tat' if retaliated else 'probe_exploit'
            
        if self.strategy_mode == 'probe_exploit':
            # Defect every 3rd move, otherwise play TFT
            if len(self.history) % 3 == 2:  # Every 3rd move (0-indexed)
                return DEFECT
            else:
                return self.opponent_history[-1]
        else:
            # Standard Tit for Tat
            return self.opponent_history[-1]


class ForgivingTitForTat(Strategy):
    """TFT but with 20% chance to forgive after 2+ consecutive defections."""
    
    def __init__(self):
        super().__init__("Forgiving Tit for Tat")
        
    def play(self) -> str:
        if not self.opponent_history:
            return COOPERATE
            
        last_move = self.opponent_history[-1]
        
        # Check for consecutive defections
        if len(self.opponent_history) >= 2:
            consecutive_defects = 0
            for move in reversed(self.opponent_history):
                if move == DEFECT:
                    consecutive_defects += 1
                else:
                    break
                    
            # 20% chance to forgive after 2+ consecutive defections
            if consecutive_defects >= 2 and random.random() < 0.2:
                return COOPERATE
                
        return last_move


class Spiteful(Strategy):
    """Cooperate until opponent defects, then play randomly with 70% defection rate."""
    
    def __init__(self):
        super().__init__("Spiteful")
        self.spite_triggered = False
        
    def reset(self):
        super().reset()
        self.spite_triggered = False
        
    def play(self) -> str:
        if not self.spite_triggered and DEFECT in self.opponent_history:
            self.spite_triggered = True
            
        if self.spite_triggered:
            # 70% chance to defect when spiteful
            return DEFECT if random.random() < 0.7 else COOPERATE
        else:
            return COOPERATE


class Bully(Strategy):
    """Defect until opponent defects back, then mostly cooperate."""
    
    def __init__(self):
        super().__init__("Bully")
        self.opponent_fought_back = False
        
    def reset(self):
        super().reset()
        self.opponent_fought_back = False
        
    def play(self) -> str:
        if not self.opponent_fought_back and DEFECT in self.opponent_history:
            self.opponent_fought_back = True
            
        if self.opponent_fought_back:
            # 80% chance to cooperate after opponent fights back
            return COOPERATE if random.random() < 0.8 else DEFECT
        else:
            # Keep bullying until opponent fights back
            return DEFECT


class RemorsefulProber(Strategy):
    """Like Prober but always cooperates twice after defecting (showing remorse)."""
    
    def __init__(self):
        super().__init__("Remorseful Prober")
        self.probe_pattern = [DEFECT, COOPERATE, COOPERATE]
        self.strategy_mode = None  # 'probe_exploit' or 'tit_for_tat'
        self.remorse_counter = 0  # Tracks consecutive cooperations after defection
        
    def reset(self):
        super().reset()
        self.strategy_mode = None
        self.remorse_counter = 0
        
    def play(self) -> str:
        # First 3 moves: follow probe pattern
        if len(self.history) < 3:
            return self.probe_pattern[len(self.history)]
            
        # After probe, determine strategy
        if self.strategy_mode is None:
            # Check if opponent retaliated to the initial defection
            retaliated = len(self.opponent_history) > 0 and self.opponent_history[0] == DEFECT
            self.strategy_mode = 'tit_for_tat' if retaliated else 'probe_exploit'
            
        # Handle remorse: cooperate twice after any defection
        if self.remorse_counter > 0:
            self.remorse_counter -= 1
            return COOPERATE
            
        if self.strategy_mode == 'probe_exploit':
            if len(self.history) % 3 == 2:  # Defect every 3rd move (0-indexed)
                self.remorse_counter = 2  # Set up 2 remorseful cooperations
                return DEFECT
            else:
                return self.opponent_history[-1]
        else:
            # Standard Tit for Tat with remorse
            if self.opponent_history[-1] == DEFECT:
                self.remorse_counter = 1  # Show remorse after retaliating
                return DEFECT
            return COOPERATE


class Tournament:
    """Manages the tournament between strategies."""
    
    def __init__(self, strategies: List[Strategy]):
        self.strategies = strategies
        self.results = defaultdict(lambda: {'total_score': 0, 'total_games': 0})
        self.match_histories = {}  # Store complete histories for each match
        
    def play_game(self, strategy1: Strategy, strategy2: Strategy) -> Tuple[int, int]:
        """Play one game between two strategies and return scores."""
        move1 = strategy1.play()
        move2 = strategy2.play()
        
        payoff1, payoff2 = PAYOFF_MATRIX[(move1, move2)]
        
        strategy1.update_history(move1, move2, payoff1)
        strategy2.update_history(move2, move1, payoff2)
        
        return payoff1, payoff2
        
    def play_match(self, strategy1: Strategy, strategy2: Strategy) -> Tuple[int, int, int]:
        """Play one match between two strategies. Returns scores and game count."""
        strategy1.reset()
        strategy2.reset()
        
        score1, score2 = 0, 0
        games = 0
        
        # Continue until random termination
        while True:
            p1, p2 = self.play_game(strategy1, strategy2)
            score1 += p1
            score2 += p2
            games += 1
            
            # Check for match termination
            if random.random() < MATCH_END_PROBABILITY:
                break
                
        return score1, score2, games
        
    def run_tournament(self):
        """Run the complete tournament."""
        print("Running Axelrod's Iterated Prisoner's Dilemma Tournament...")
        print(f"Each pair plays {MATCHES_PER_PAIR} matches")
        print(f"Expected games per match: ~{int(1/MATCH_END_PROBABILITY)}")
        print("-" * 60)
        
        # Calculate total number of matches for progress tracking
        n_strategies = len(self.strategies)
        total_matches = 0
        for i in range(n_strategies):
            for j in range(i + 1, n_strategies):  # Only count unique pairs (no self-play)
                total_matches += MATCHES_PER_PAIR
        
        print(f"Total matches to play: {total_matches:,}")
        print(f"Strategy pairings: {n_strategies * (n_strategies - 1) // 2}")
        print("-" * 60)
        
        # Initialize progress bar
        pbar = tqdm(total=total_matches, 
                   desc="Tournament Progress", 
                   unit="match",
                   ncols=100)
        
        match_count = 0
        
        # Play all pairwise matches
        for i, strat1 in enumerate(self.strategies):
            for j, strat2 in enumerate(self.strategies):
                if i < j:  # Play against others only
                    for match_num in range(MATCHES_PER_PAIR):
                        score1, score2, games = self.play_match(strat1, strat2)
                        
                        # Update results for strategy 1
                        self.results[strat1.name]['total_score'] += score1
                        self.results[strat1.name]['total_games'] += games
                        
                        # Update results for strategy 2
                        self.results[strat2.name]['total_score'] += score2
                        self.results[strat2.name]['total_games'] += games
                            
                        # Store match history
                        match_key = f"{strat1.name} vs {strat2.name} (Match {match_num + 1})"
                        self.match_histories[match_key] = {
                            'player1_moves': strat1.history.copy(),
                            'player2_moves': strat2.history.copy(),
                            'games': games,
                            'scores': (score1, score2)
                        }
                        
                        # Update progress bar
                        match_count += 1
                        pbar.update(1)
                        
                        # Update description with current matchup every 100 matches
                        if match_count % 100 == 0:
                            pbar.set_description(f"Playing: {strat1.name[:12]} vs {strat2.name[:12]}")
        
        pbar.close()
        print("\nTournament completed!")
                        
    def display_results(self):
        """Display tournament results sorted by average score per game."""
        print("\nTOURNAMENT RESULTS")
        print("=" * 70)
        print(f"{'Rank':<6}{'Strategy':<25}{'Avg Score/Game':<18}{'Total Games':<12}")
        print("-" * 70)
        
        # Calculate average scores and sort
        rankings = []
        for name, data in self.results.items():
            avg_score = data['total_score'] / data['total_games']
            rankings.append((name, avg_score, data['total_games']))
            
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Display rankings
        for rank, (name, avg_score, total_games) in enumerate(rankings, 1):
            print(f"{rank:<6}{name:<25}{avg_score:<18.3f}{total_games:<12}")
            
        print("=" * 70)
        
        # Display some interesting statistics
        print("\nADDITIONAL STATISTICS:")
        print(f"Total matches played: {len(self.match_histories)}")
        
        total_games_all = sum(hist['games'] for hist in self.match_histories.values())
        avg_games_per_match = total_games_all / len(self.match_histories)
        print(f"Average games per match: {avg_games_per_match:.1f}")
        
        # Find longest and shortest matches
        longest = max(self.match_histories.items(), key=lambda x: x[1]['games'])
        shortest = min(self.match_histories.items(), key=lambda x: x[1]['games'])
        print(f"Longest match: {longest[0]} ({longest[1]['games']} games)")
        print(f"Shortest match: {shortest[0]} ({shortest[1]['games']} games)")


def main():
    """Run the tournament."""
    # Create all strategies
    strategies = [
        TitForTat(),
        TitForTwoTats(),
        GenerousTitForTat(),
        Pavlov(),
        AlwaysDefect(),
        AlwaysCooperate(),
        GrimTrigger(),
        Tester(),
        Joss(),
        RandomStrategy(),
        Grudger(),
        SuspiciousTitForTat(),
        Alternator(),
        Adaptive(),
        Detective(),
        Prober(),
        ForgivingTitForTat(),
        Spiteful(),
        Bully(),
        RemorsefulProber()
    ]
    
    # Try to add DeepLearningStrategy if model exists
    if os.path.exists("prisoner_dilemma_ppo_model.zip"):
        try:
            from train_dl_strategy import DeepLearningStrategy
            strategies.append(DeepLearningStrategy())
            print("Added Deep Learning strategy to tournament")
        except Exception as e:
            print(f"Could not load Deep Learning strategy: {e}")
    else:
        print("Deep Learning model not found. Run dl_strategy.py first to train it.")
    
    # Run tournament
    tournament = Tournament(strategies)
    tournament.run_tournament()
    tournament.display_results()


if __name__ == "__main__":
    main()