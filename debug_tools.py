"""
Debug tools for Cho-Pha-Go AlphaZero implementation

This module provides utilities for debugging and validating the AlphaZero implementation
for 5x5 Go, focusing on finding potential logical errors in the MCTS, neural network,
and game rule implementations.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from minigo.minigo import Position, BLACK, WHITE
from minigo.features import to_default_tensor
from cho_pha_go_train import MCTSNode, AlphaGoZeroNet
import copy
import os
import json
from collections import defaultdict

class AlphaZeroDebugger:
    """Tools for debugging and validating AlphaZero components."""
    
    def __init__(self, model_path=None, board_size=5, device='cpu'):
        """
        Initialize the debugger with a model.
        
        Args:
            model_path: Path to a trained model
            board_size: Size of the Go board
            device: Device to run the model on
        """
        self.board_size = board_size
        self.device = device
        
        # Initialize model
        self.model = AlphaGoZeroNet(board_size=board_size)
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.load(model_path, device)
        self.model.to(device)
        
        # Create directory for debug outputs
        os.makedirs('debug', exist_ok=True)
    
    def validate_mcts_search(self, position=None, num_simulations=800, 
                            network_trust=0.5, exploration=1.4, noise_level=0.003,
                            log_file='debug/mcts_log.txt'):
        """
        Validate MCTS search by running it with detailed logging.
        
        Args:
            position: Starting position (default: empty board)
            num_simulations: Number of MCTS simulations
            network_trust: Weight given to the neural network's policy
            exploration: Exploration constant for UCB
            noise_level: Dirichlet noise level
            log_file: File to write logs to
        """
        if position is None:
            position = Position()  # Empty board, black to play
        
        with open(log_file, 'w') as f:
            f.write(f"=== MCTS Validation with {num_simulations} simulations ===\n")
            f.write(f"Starting position:\n{position}\n")
            f.write(f"Parameters: network_trust={network_trust}, exploration={exploration}, noise_level={noise_level}\n\n")
            
            # Run MCTS with verbose output
            root = MCTSNode(position, exploration=exploration)
            
            # Get policy from neural network
            state_tensor = to_default_tensor(position).to(self.model.device)
            policy, value = self.model(state_tensor)
            policy = policy.detach().cpu().numpy()[0]
            value = value.detach().cpu().numpy()[0][0]
            
            f.write(f"Neural network initial evaluation:\n")
            f.write(f"Value: {value:.4f}\n")
            
            # Log policy distribution for legal moves
            f.write("Initial policy distribution:\n")
            for move in position.all_legal_moves():
                if move is None:  # Pass move
                    move_idx = self.board_size * self.board_size
                else:
                    move_idx = move[0] * self.board_size + move[1]
                f.write(f"  {move}: {policy[move_idx]:.4f}\n")
            
            f.write("\nRunning MCTS simulations...\n")
            
            # Run MCTS search
            visit_counts = defaultdict(list)
            q_values = defaultdict(list)
            
            # Sample at different points
            sample_points = [50, 100, 200, 400, 800] if num_simulations >= 800 else \
                            [int(num_simulations * frac) for frac in [0.1, 0.25, 0.5, 0.75, 1.0]]
            sample_points = [p for p in sample_points if p <= num_simulations]
            
            for i in range(1, num_simulations + 1):
                # Select a leaf node
                node = root
                search_path = [node]
                
                while node.children and not node.position.is_game_over():
                    node = node.select_child(noise_level if node is root else 0)
                    search_path.append(node)
                
                # Expand the leaf node if it's not terminal
                if not node.position.is_game_over():
                    state_tensor = to_default_tensor(node.position).to(self.model.device)
                    policy, value = self.model(state_tensor)
                    policy = policy.detach().cpu().numpy()[0]
                    value = value.detach().cpu().numpy()[0][0]
                    
                    # Apply network trust
                    if node.children:
                        prior_score = network_trust * value + (1-network_trust) * node.value
                    else:
                        node.expand(policy)
                        prior_score = value
                else:
                    # Terminal node
                    prior_score = node.position.result()
                
                # Backpropagate
                for node in reversed(search_path):
                    node.update(prior_score)
                    prior_score = -prior_score  # Flip for opponent's perspective
                
                # Log at sample points
                if i in sample_points:
                    f.write(f"\nAfter {i} simulations:\n")
                    for action, child in root.children.items():
                        visit_count = child.visits
                        q_value = child.value
                        visit_counts[action].append(visit_count)
                        q_values[action].append(q_value)
                        
                        f.write(f"  {action}: visits={visit_count}, Q={q_value:.4f}, " + 
                                f"UCB={root._ucb_score(child):.4f}\n")
            
            # Choose best action
            action, probs = root.best_action(temperature=1.0)
            f.write(f"\nFinal selection: {action}\n")
            f.write(f"Final move probabilities: {probs}\n")
            
            # Plot visit count evolution
            plt.figure(figsize=(10, 6))
            for action in visit_counts:
                if action is None:
                    label = "Pass"
                else:
                    label = f"({action[0]},{action[1]})"
                plt.plot(sample_points, visit_counts[action], marker='o', label=label)
            
            plt.title("MCTS Visit Count Evolution")
            plt.xlabel("Simulations")
            plt.ylabel("Visit Count")
            plt.legend()
            plt.grid(True)
            plt.savefig('debug/visit_counts.png')
            
            # Plot Q-value evolution
            plt.figure(figsize=(10, 6))
            for action in q_values:
                if action is None:
                    label = "Pass"
                else:
                    label = f"({action[0]},{action[1]})"
                plt.plot(sample_points, q_values[action], marker='o', label=label)
            
            plt.title("MCTS Q-Value Evolution")
            plt.xlabel("Simulations")
            plt.ylabel("Q-Value")
            plt.legend()
            plt.grid(True)
            plt.savefig('debug/q_values.png')
            
            f.write("\nDebug plots saved to debug/visit_counts.png and debug/q_values.png\n")
            
            return root, action, probs
    
    def validate_center_advantage(self, num_games=100, num_simulations=400):
        """
        Test if the model recognizes the center advantage for black.
        
        Args:
            num_games: Number of games to play
            num_simulations: Number of MCTS simulations per move
            
        Returns:
            Dictionary with statistics about center play frequency and win rates
        """
        stats = {
            'black_center_first_count': 0,
            'black_center_first_wins': 0,
            'black_other_first_count': 0,
            'black_other_first_wins': 0
        }
        
        for game_idx in range(num_games):
            # Start a new game
            position = Position()
            game_history = []
            
            # Play until game is over
            while not position.is_game_over():
                # Run MCTS to select a move
                root = MCTSNode(position, exploration=1.4)
                root = root.search(self.model, position, num_simulations=num_simulations, 
                                  network_trust=0.5, noise_level=0.001)
                
                # Choose action based on visit count
                action, _ = root.best_action(temperature=1.0)
                
                # Record first move by black
                if len(game_history) == 0:  # First move
                    if action == (2, 2):  # Center
                        stats['black_center_first_count'] += 1
                    else:
                        stats['black_other_first_count'] += 1
                
                # Apply the move
                game_history.append((position.to_play, action))
                position = position.play_move(action, color=position.to_play)
            
            # Record game result
            result = position.result()
            
            if game_history[0][1] == (2, 2):  # Black played center first
                if result == BLACK:
                    stats['black_center_first_wins'] += 1
            else:  # Black didn't play center first
                if result == BLACK:
                    stats['black_other_first_wins'] += 1
            
            # Print progress
            if (game_idx + 1) % 10 == 0:
                print(f"Completed {game_idx + 1}/{num_games} test games")
        
        # Calculate statistics
        if stats['black_center_first_count'] > 0:
            stats['black_center_first_win_rate'] = stats['black_center_first_wins'] / stats['black_center_first_count']
        else:
            stats['black_center_first_win_rate'] = 0
            
        if stats['black_other_first_count'] > 0:
            stats['black_other_first_win_rate'] = stats['black_other_first_wins'] / stats['black_other_first_count']
        else:
            stats['black_other_first_win_rate'] = 0
        
        stats['center_first_preference'] = stats['black_center_first_count'] / num_games
        
        # Save stats to file
        with open('debug/center_advantage_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"Center play frequency: {stats['center_first_preference']:.2f}")
        print(f"Center-first win rate: {stats['black_center_first_win_rate']:.2f}")
        print(f"Other-first win rate: {stats['black_other_first_win_rate']:.2f}")
        
        return stats
    
    def analyze_policy_grid(self, position=None):
        """
        Visualize the policy network output for a given position.
        
        Args:
            position: Go position to analyze (default: empty board)
            
        Returns:
            Policy grid visualization saved to debug/policy_grid.png
        """
        if position is None:
            position = Position()  # Empty board
        
        # Get policy from neural network
        state_tensor = to_default_tensor(position).to(self.model.device)
        policy, value = self.model(state_tensor)
        policy = policy.detach().cpu().numpy()[0]
        
        # Reshape policy (excluding pass move)
        policy_grid = policy[:-1].reshape(self.board_size, self.board_size)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(policy_grid, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Move Probability')
        
        # Add text annotations
        for i in range(self.board_size):
            for j in range(self.board_size):
                plt.text(j, i, f"{policy_grid[i, j]:.3f}", 
                         ha="center", va="center", color="w" if policy_grid[i, j] < 0.4 else "k")
        
        # Add board position overlay
        for i in range(self.board_size):
            for j in range(self.board_size):
                if position.board[i, j] == BLACK:
                    plt.plot(j, i, 'o', markersize=15, markerfacecolor='k')
                elif position.board[i, j] == WHITE:
                    plt.plot(j, i, 'o', markersize=15, markerfacecolor='w', markeredgecolor='k')
        
        plt.grid(color='black', linestyle='-', linewidth=1)
        plt.xticks(range(self.board_size))
        plt.yticks(range(self.board_size))
        plt.title(f"Policy Network Output (Value: {value.item():.3f})")
        plt.savefig('debug/policy_grid.png')
        
        print(f"Policy grid visualization saved to debug/policy_grid.png")
        return policy_grid

    def debug_game_progression(self, start_position=None, moves=None, num_simulations=400):
        """
        Debug a specific game progression to analyze model behavior.
        
        Args:
            start_position: Starting position (default: empty board)
            moves: List of moves to apply in sequence, or None to use MCTS
            num_simulations: Number of MCTS simulations per move if moves=None
            
        Returns:
            Dictionary with data about the game progression
        """
        if start_position is None:
            position = Position()  # Empty board
        else:
            position = copy.deepcopy(start_position)
            
        game_data = {
            'positions': [],
            'policies': [],
            'values': [],
            'moves': [],
            'visit_counts': []
        }
        
        # Record initial position
        game_data['positions'].append(str(position))
        
        # Get initial policy and value
        state_tensor = to_default_tensor(position).to(self.model.device)
        policy, value = self.model(state_tensor)
        game_data['policies'].append(policy.detach().cpu().numpy()[0].tolist())
        game_data['values'].append(value.item())
        
        if moves is None:
            # Play using MCTS
            while not position.is_game_over():
                root = MCTSNode(position, exploration=1.4)
                root = root.search(self.model, position, num_simulations=num_simulations, 
                                  network_trust=0.5, noise_level=0.001)
                
                # Record visit counts
                visit_count = {str(action): child.visits for action, child in root.children.items()}
                game_data['visit_counts'].append(visit_count)
                
                # Choose action
                action, _ = root.best_action(temperature=1.0)
                game_data['moves'].append(str(action))
                
                # Apply move
                position = position.play_move(action, color=position.to_play)
                
                # Record new position
                game_data['positions'].append(str(position))
                
                # Get policy and value for new position
                if not position.is_game_over():
                    state_tensor = to_default_tensor(position).to(self.model.device)
                    policy, value = self.model(state_tensor)
                    game_data['policies'].append(policy.detach().cpu().numpy()[0].tolist())
                    game_data['values'].append(value.item())
        else:
            # Play specified moves
            for move in moves:
                game_data['moves'].append(str(move))
                
                # Apply move
                position = position.play_move(move, color=position.to_play)
                
                # Record new position
                game_data['positions'].append(str(position))
                
                # Get policy and value for new position
                if not position.is_game_over():
                    state_tensor = to_default_tensor(position).to(self.model.device)
                    policy, value = self.model(state_tensor)
                    game_data['policies'].append(policy.detach().cpu().numpy()[0].tolist())
                    game_data['values'].append(value.item())
        
        # Save game data
        with open('debug/game_progression.json', 'w') as f:
            json.dump(game_data, f, indent=2)
            
        # Create value progression plot
        plt.figure(figsize=(10, 6))
        plt.plot(game_data['values'], marker='o')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.grid(True)
        plt.title("Value Network Predictions During Game")
        plt.xlabel("Move Number")
        plt.ylabel("Value (from black's perspective)")
        plt.savefig('debug/value_progression.png')
        
        print(f"Game progression data saved to debug/game_progression.json")
        print(f"Value progression plot saved to debug/value_progression.png")
        
        return game_data


if __name__ == "__main__":
    debugger = AlphaZeroDebugger(model_path='models/cho_pha_go_5x5.pt')
    
    # Validate MCTS search
    print("Validating MCTS search...")
    debugger.validate_mcts_search(num_simulations=400)
    
    # Test center advantage
    print("\nTesting center advantage recognition...")
    debugger.validate_center_advantage(num_games=20, num_simulations=400)
    
    # Analyze policy grid for empty board
    print("\nAnalyzing policy grid for empty board...")
    debugger.analyze_policy_grid()
    
    # Debug game progression
    print("\nDebugging game progression...")
    # Scenario 1: Black plays center first
    center_game = debugger.debug_game_progression(
        moves=[(2, 2), (1, 1), (2, 1), (1, 2), (3, 2), (1, 3), (2, 3)]
    )
