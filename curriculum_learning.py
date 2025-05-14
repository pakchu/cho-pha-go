"""
Curriculum learning module for AlphaZero 5x5 Go training

This module implements curriculum learning strategies to help the model converge
to the theoretically optimal play (center-first) for 5x5 Go.
"""

import numpy as np
import os
import torch
from minigo.minigo import Position, BLACK, WHITE
from minigo.features import to_default_tensor
from cho_pha_go_train import ReplayBuffer

class CurriculumLearning:
    """Curriculum learning strategies for 5x5 Go"""
    
    def __init__(self, replay_buffer=None, capacity=5000, device='cpu'):
        """
        Initialize the curriculum learning module.
        
        Args:
            replay_buffer: An existing replay buffer or None to create a new one
            capacity: Capacity of the replay buffer if creating a new one
            device: Device to use for tensor operations
        """
        self.device = device
        if replay_buffer is None:
            self.replay_buffer = ReplayBuffer(capacity=capacity, device=device)
        else:
            self.replay_buffer = replay_buffer
    
    def generate_center_curriculum(self, num_positions=1000, win_rate=0.85):
        """
        Generate a curriculum of positions that emphasize the center-first advantage.
        
        Args:
            num_positions: Number of positions to generate
            win_rate: Target win rate for black in the curriculum
            
        Returns:
            Number of positions added to the replay buffer
        """
        positions_added = 0
        
        # 1. Generate winning positions for black with center stone
        num_black_wins = int(num_positions * win_rate)
        for _ in range(num_black_wins):
            # Create a random board state where black has played center and is winning
            board = np.zeros((5, 5), dtype=np.int8)
            
            # Always place center stone for black
            board[2, 2] = BLACK
            
            # Add some surrounding black stones (2-5 stones)
            potential_black_positions = [
                (1, 1), (1, 2), (1, 3), 
                (2, 1), (2, 3), 
                (3, 1), (3, 2), (3, 3)
            ]
            
            num_extra_black = np.random.randint(2, 6)
            black_indices = np.random.choice(len(potential_black_positions), num_extra_black, replace=False)
            
            for idx in black_indices:
                x, y = potential_black_positions[idx]
                board[x, y] = BLACK
            
            # Add some white stones (fewer than black)
            potential_white_positions = [
                (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 0), (1, 4),
                (2, 0), (2, 4),
                (3, 0), (3, 4),
                (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)
            ]
            
            # Filter out positions that are already occupied
            potential_white_positions = [(x, y) for x, y in potential_white_positions 
                                         if board[x, y] == 0]
            
            num_white = np.random.randint(2, 5)
            if len(potential_white_positions) > 0:
                white_indices = np.random.choice(len(potential_white_positions), 
                                               min(num_white, len(potential_white_positions)), 
                                               replace=False)
                
                for idx in white_indices:
                    x, y = potential_white_positions[idx]
                    board[x, y] = WHITE
            
            # Create position and add to replay buffer
            position = Position(board=board.copy())
            self._add_position_to_buffer(position, result=BLACK)
            positions_added += 1
        
        # 2. Generate some positions where black doesn't play center and loses
        num_black_losses = num_positions - num_black_wins
        for _ in range(num_black_losses):
            board = np.zeros((5, 5), dtype=np.int8)
            
            # White takes center
            board[2, 2] = WHITE
            
            # Place black stones away from center
            potential_black_positions = [
                (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 0), (1, 4),
                (2, 0), (2, 4),
                (3, 0), (3, 4),
                (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)
            ]
            
            num_black = np.random.randint(3, 6)
            black_indices = np.random.choice(len(potential_black_positions), num_black, replace=False)
            
            for idx in black_indices:
                x, y = potential_black_positions[idx]
                board[x, y] = BLACK
            
            # Add some more white stones around center
            potential_white_positions = [
                (1, 1), (1, 2), (1, 3), 
                (2, 1), (2, 3), 
                (3, 1), (3, 2), (3, 3)
            ]
            
            # Filter out positions that are already occupied
            potential_white_positions = [(x, y) for x, y in potential_white_positions 
                                         if board[x, y] == 0]
            
            num_extra_white = np.random.randint(2, 5)
            if len(potential_white_positions) > 0:
                white_indices = np.random.choice(len(potential_white_positions), 
                                               min(num_extra_white, len(potential_white_positions)), 
                                               replace=False)
                
                for idx in white_indices:
                    x, y = potential_white_positions[idx]
                    board[x, y] = WHITE
            
            # Create position and add to replay buffer
            position = Position(board=board.copy())
            self._add_position_to_buffer(position, result=WHITE)
            positions_added += 1
            
        print(f"Added {positions_added} curriculum positions to replay buffer")
        return positions_added
    
    def generate_opening_book(self, num_positions=100):
        """
        Generate opening positions with strong emphasis on center-first play.
        
        Args:
            num_positions: Number of opening positions to generate
            
        Returns:
            Number of positions added to replay buffer
        """
        positions_added = 0
        
        # 1. Empty board positions with strong policy for center
        for _ in range(num_positions // 4):
            position = Position()  # Empty board
            
            # Create policy with high probability for center move
            policy = np.ones(5*5 + 1) * 0.01  # Small probability for all moves
            center_idx = 2*5 + 2  # Center position index
            policy[center_idx] = 0.8  # 80% probability for center
            policy = policy / policy.sum()  # Normalize
            
            # Add to replay buffer (assuming black wins)
            state_tensor = to_default_tensor(position).to(self.device)
            self.replay_buffer.buffer.append((state_tensor, torch.FloatTensor(policy), 1))
            positions_added += 1
        
        # 2. First move center by black with various second moves by white
        for _ in range(num_positions // 4):
            board = np.zeros((5, 5), dtype=np.int8)
            board[2, 2] = BLACK  # Center first for black
            
            # Choose random second move for white
            while True:
                wx, wy = np.random.randint(0, 5, size=2)
                if board[wx, wy] == 0:
                    board[wx, wy] = WHITE
                    break
            
            position = Position(board=board.copy(), to_play=BLACK)
            
            # Create a policy that encourages black to play near center
            policy = np.ones(5*5 + 1) * 0.01
            
            # Increase probability for moves adjacent to center
            adjacent_to_center = [(1, 2), (2, 1), (2, 3), (3, 2)]
            for x, y in adjacent_to_center:
                if position.board[x, y] == 0:  # If empty
                    move_idx = x*5 + y
                    policy[move_idx] = 0.2
            
            policy = policy / policy.sum()  # Normalize
            
            # Add to replay buffer (assuming black wins)
            state_tensor = to_default_tensor(position).to(self.device)
            self.replay_buffer.buffer.append((state_tensor, torch.FloatTensor(policy), 1))
            positions_added += 1
        
        # 3. First few moves with strong center control by black
        for _ in range(num_positions // 2):
            board = np.zeros((5, 5), dtype=np.int8)
            
            # Black's first move at center
            board[2, 2] = BLACK
            
            # White's response (not center)
            while True:
                wx, wy = np.random.randint(0, 5, size=2)
                if board[wx, wy] == 0:
                    board[wx, wy] = WHITE
                    break
            
            # Black's second move - adjacent to center
            adjacent_to_center = [(1, 2), (2, 1), (2, 3), (3, 2)]
            bx, by = adjacent_to_center[np.random.randint(0, len(adjacent_to_center))]
            board[bx, by] = BLACK
            
            # White's second move
            empty_positions = [(x, y) for x in range(5) for y in range(5) if board[x, y] == 0]
            if empty_positions:
                wx, wy = empty_positions[np.random.randint(0, len(empty_positions))]
                board[wx, wy] = WHITE
            
            # Create position and policy
            position = Position(board=board.copy(), to_play=BLACK)
            
            # Simple policy for next good moves
            policy = np.ones(5*5 + 1) * 0.01
            
            # Find empty positions adjacent to black stones
            for x in range(5):
                for y in range(5):
                    if board[x, y] == BLACK:
                        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                            if 0 <= nx < 5 and 0 <= ny < 5 and board[nx, ny] == 0:
                                move_idx = nx*5 + ny
                                policy[move_idx] += 0.1
            
            policy = policy / policy.sum()  # Normalize
            
            # Add to replay buffer (assuming black wins)
            state_tensor = to_default_tensor(position).to(self.device)
            self.replay_buffer.buffer.append((state_tensor, torch.FloatTensor(policy), 1))
            positions_added += 1
        
        print(f"Added {positions_added} opening book positions to replay buffer")
        return positions_added
    
    def _add_position_to_buffer(self, position, result):
        """Helper method to add a position to the replay buffer"""
        # Create policy based on current position
        policy = np.ones(5*5 + 1) * (1.0 / (5*5 + 1))  # Uniform policy
        
        # If it's black's turn in a black win position, give high probability to good moves
        if position.to_play == BLACK and result == BLACK:
            # Find empty positions that are adjacent to black stones
            for x in range(5):
                for y in range(5):
                    if position.board[x, y] == 0:
                        # Check if adjacent to a black stone
                        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                            if 0 <= nx < 5 and 0 <= ny < 5 and position.board[nx, ny] == BLACK:
                                move_idx = x*5 + y
                                policy[move_idx] += 0.1
                                break
            
            # If center is empty, give it high probability
            if position.board[2, 2] == 0:
                policy[2*5 + 2] = 0.5
        
        # If it's white's turn in a black win position, defensive moves
        elif position.to_play == WHITE and result == BLACK:
            # Try to block black's expansion
            for x in range(5):
                for y in range(5):
                    if position.board[x, y] == 0:
                        # Check if adjacent to a black stone
                        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                            if 0 <= nx < 5 and 0 <= ny < 5 and position.board[nx, ny] == BLACK:
                                move_idx = x*5 + y
                                policy[move_idx] += 0.1
                                break
        
        # Similar logic for white's win positions
        elif position.to_play == WHITE and result == WHITE:
            for x in range(5):
                for y in range(5):
                    if position.board[x, y] == 0:
                        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                            if 0 <= nx < 5 and 0 <= ny < 5 and position.board[nx, ny] == WHITE:
                                move_idx = x*5 + y
                                policy[move_idx] += 0.1
                                break
                                
        # Normalize policy
        policy = policy / policy.sum()
        
        # Convert result to value (-1, 0, 1)
        value = 1 if result == BLACK else -1 if result == WHITE else 0
        
        # Add to replay buffer
        state_tensor = to_default_tensor(position).to(self.device)
        self.replay_buffer.buffer.append((state_tensor, torch.FloatTensor(policy), value))
    
    def create_progressive_schedule(self, num_iterations=100):
        """
        Create a progressive hyperparameter schedule for training.
        
        Args:
            num_iterations: Total number of training iterations
            
        Returns:
            List of (iteration, network_trust, exploration, temperature, noise_level) tuples
        """
        schedule = []
        
        # Phase 1: High exploration, low network trust (first 20%)
        phase1_end = int(num_iterations * 0.2)
        for it in range(phase1_end):
            network_trust = 0.1 + (it / phase1_end) * 0.3  # 0.1 to 0.4
            exploration = 1.4 - (it / phase1_end) * 0.4    # 1.4 to 1.0
            temperature = 1.0
            noise_level = 0.03 - (it / phase1_end) * 0.02  # 0.03 to 0.01
            schedule.append((it, network_trust, exploration, temperature, noise_level))
        
        # Phase 2: Balanced exploration/exploitation (next 40%)
        phase2_end = int(num_iterations * 0.6)
        for it in range(phase1_end, phase2_end):
            progress = (it - phase1_end) / (phase2_end - phase1_end)
            network_trust = 0.4 + progress * 0.3  # 0.4 to 0.7
            exploration = 1.0 - progress * 0.4    # 1.0 to 0.6
            temperature = 0.8 - progress * 0.3    # 0.8 to 0.5
            noise_level = 0.01 - progress * 0.005  # 0.01 to 0.005
            schedule.append((it, network_trust, exploration, temperature, noise_level))
        
        # Phase 3: High network trust, low exploration (final 40%)
        for it in range(phase2_end, num_iterations):
            progress = (it - phase2_end) / (num_iterations - phase2_end)
            network_trust = 0.7 + progress * 0.3  # 0.7 to 1.0
            exploration = 0.6 - progress * 0.3    # 0.6 to 0.3
            temperature = 0.5 - progress * 0.4    # 0.5 to 0.1
            noise_level = 0.005 - progress * 0.003  # 0.005 to 0.002
            schedule.append((it, network_trust, exploration, temperature, noise_level))
        
        return schedule

if __name__ == "__main__":
    # Example usage
    curriculum = CurriculumLearning(device='cpu')
    
    # Generate curriculum positions
    curriculum.generate_center_curriculum(num_positions=500)
    
    # Generate opening book
    curriculum.generate_opening_book(num_positions=200)
    
    # Print schedule example
    schedule = curriculum.create_progressive_schedule(num_iterations=50)
    print("\nProgressive Hyperparameter Schedule (sample):")
    for i, params in enumerate(schedule):
        if i % 10 == 0 or i == len(schedule) - 1:
            it, trust, explore, temp, noise = params
            print(f"Iteration {it}: trust={trust:.2f}, exploration={explore:.2f}, temp={temp:.2f}, noise={noise:.4f}")
