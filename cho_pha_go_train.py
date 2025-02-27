import copy
import os
import multiprocessing as mp
import numpy as np
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from go.go_board import FastState as State
import warnings
from utils import timeit

warnings.filterwarnings("ignore", category=UserWarning)

#####################################
# 1. 알파고 제로 신경망 정의
#####################################
class AlphaGoZeroNet(nn.Module):
    def __init__(self, board_size=19, verbose=False):
        super(AlphaGoZeroNet, self).__init__()
        self.device = None
        self.board_size = board_size
        self.losses = defaultdict(list)
        self.optimizer = None
        self.verbose = verbose
        # 네트워크 크기 동적 조정
        if board_size <= 9:
            self.num_filters = 64  # 작은 보드에서는 필터 수를 줄임
            self.num_residual_blocks = 5  # Residual Block 개수 축소
        elif board_size <= 13:
            self.num_filters = 128
            self.num_residual_blocks = 12
        else:
            self.num_filters = 256  # 기본 크기
            self.num_residual_blocks = 19

        # 첫 Conv (입력 채널: 17)
        self.conv1 = nn.Conv2d(17, self.num_filters, kernel_size=3, padding=1)

        # Residual Block 정의
        self.residual_blocks = nn.ModuleList(
            [self._build_residual_block() for _ in range(self.num_residual_blocks)]
        )

        # Policy Head: 출력 차원은 board_size^2 + 1 (마지막 하나가 pass)
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size * 2, board_size * board_size + 1),
            nn.Softmax(dim=1)
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(self.num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def _build_residual_block(self):
        """Residual Block 정의"""
        return nn.Sequential(
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_filters)
        )

    def forward(self, x):
        """모델 순전파"""
        x = F.relu(self.conv1(x))
        for block in self.residual_blocks:
            skip = x
            x = block(x)
            x = F.relu(x + skip)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = self.get_device()
        return self

    def get_device(self):
        # 파라미터 또는 버퍼에서 device 추출
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    def copy(self):
        """모델 복사"""
        model = AlphaGoZeroNet(board_size=self.board_size)
        model.load_state_dict(self.state_dict())
        model.device = self.device
        model.losses = copy.copy(self.losses)
        return model

    def save(self, path):
        tail = f'{self.board_size}x{self.board_size}' in path
        path = path + f'_{self.board_size}x{self.board_size}.pt' if not tail else path
        torch.save({
            'model_state_dict': self.state_dict(),
            'losses': self.losses
        }, path)
        print(f"Model saved to {path}.")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.losses = checkpoint['losses']
        print(f"Model loaded from {path}.")

    def mcts_search(self, state: State, num_simulations=800):
        """MCTS 탐색: root 노드부터 시뮬레이션 수행"""
        root = MCTSNode(state)
        for _ in range(num_simulations):
            node = root
            # 1) Selection
            while len(node.children) > 0 and not node.state.is_terminal():
                node = node.select_child()

            # 2) Expansion
            if not node.state.is_terminal():
                with torch.no_grad():
                    state_tensor = node.state.to_tensor().to(self.device)
                    policy, value = self(state_tensor.unsqueeze(0))
                    # breakpoint()
                    policy = policy.squeeze().cpu().numpy()  # shape: (board_size^2+1,)
                    value = value.item()
                board_size = state.board.shape[0]
                pass_prob = policy[-1]      # 마지막 요소가 pass 확률
                policy = policy[:-1]        # 나머지는 착수 확률
                policy_2d = policy.reshape(board_size, board_size)
                legal_moves = node.state.get_legal_actions()
                action_probs = []
                total_prob = 0.0
                for move in legal_moves:
                    if move is None:
                        action_probs.append((None, pass_prob))
                        total_prob += pass_prob
                    else:
                        x, y = move
                        p = policy_2d[y][x]
                        action_probs.append(((x, y), p))
                        total_prob += p
                if total_prob > 0:
                    action_probs = [(a, p / total_prob) for a, p in action_probs]
                else:
                    action_probs = [(a, 1.0 / len(legal_moves)) for a in legal_moves]
                node.expand(action_probs)
            else:
                value = node.state.get_result()
            # 3) Backpropagation
            while node is not None:
                node.update(value)
                value = -value
                node = node.parent
        return root

    def make_move(self, state: State, temperature=1.0):
        """현재 상태에서 MCTS를 이용해 행동 선택"""
        self.eval()
        root = self.mcts_search(state, num_simulations=800)
        return root.best_action(temperature=temperature, verbose=self.verbose)

#####################################
# 2. MCTS 구현
#####################################
class MCTSNode:
    def __init__(self, state: State, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.policy = 0  # 이 노드(action)에 대한 사전 확률

    def expand(self, action_probs):
        """모든 합법 착수에 대해 자식 노드 확장"""
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(self.state.apply_action(action), parent=self)
                self.children[action].policy = prob

    def select_child(self, exploration_param=1.4):
        """UCB 최대화 기준으로 자식 노드 선택"""
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            score = self._ucb_score(child, exploration_param)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _ucb_score(self, child, exploration_param):
        if child.visits == 0:
            return float('inf')
        avg_value = child.value_sum / child.visits
        ucb = avg_value + exploration_param * child.policy * np.sqrt(np.log(self.visits) / child.visits)
        return ucb

    def update(self, value):
        """노드 백업: 방문 수 및 가치 합 업데이트"""
        self.visits += 1
        self.value_sum += value

    def best_action(self, temperature=1.0, verbose=False):
        """
        가장 좋은 행동 리턴.
        temperature=0이면 방문 수가 가장 많은 행동을,
        그 외에는 방문 수 비례 샘플링.
        """
        if len(self.children) == 0:
            return None

        if temperature == 0:
            return max(self.children.keys(), key=lambda a: self.children[a].visits)
        else:
            visits = np.array([child.visits for child in self.children.values()], dtype=np.float32)
            # 모든 방문 횟수가 0이면 균등 확률 분포 사용
            if np.sum(visits) == 0:
                probs = np.ones_like(visits) / len(visits)
            else:
                adjusted = visits ** (1.0 / temperature)
                probs = adjusted / np.sum(adjusted)
            # 부동소수점 오차로 인한 문제를 방지하기 위해 재정규화
            probs = probs / np.sum(probs)
            actions = list(self.children.keys())
            if verbose:
                print(f"Normalized probabilities: {probs}, Sum: {np.sum(probs)}")

            return actions[np.random.choice(len(actions), p=probs)]


#####################################
# 4. 자가 대국 (self-play)
#####################################
@timeit
def self_play(model: AlphaGoZeroNet, init_state: State, num_simulations=800, temperature=1.0, debug=False, device='cpu'):
    """
    게임이 종료될 때까지 MCTS로 행동 선택 후,
    (state_tensor, action_probs, result)를 리턴.
    """
    # model = model.to(device)
    data = []  # (state_tensor, action_probs, result)
    current_state = init_state

    while not current_state.is_terminal():
        root = model.mcts_search(current_state, num_simulations)
        visits = np.array([child.visits for child in root.children.values()], dtype=np.float32)
        actions = list(root.children.keys())
        probs = visits / np.sum(visits)

        if temperature == 0:
            action = max(root.children, key=lambda a: root.children[a].visits)
        else:
            visits = visits ** (1.0 / temperature)
            visits /= np.sum(visits)
            if len(actions) > 0:
                action = actions[np.random.choice(len(actions), p=visits)]
            else:
                if current_state.is_terminal():
                    break

        board_size = current_state.board.shape[0]
        # full_action_probs: (board_size^2 + 1) 차원으로 수정 (마지막 원소가 pass)
        full_action_probs = np.zeros((board_size * board_size + 1,), dtype=np.float32)
        for i, act in enumerate(actions):
            if act is None:
                full_action_probs[-1] = probs[i]
            else:
                x, y = act
                idx = y * board_size + x
                full_action_probs[idx] = probs[i]

        state_tensor = current_state.to_tensor()  # shape: (17, board_size, board_size)
        data.append((state_tensor, full_action_probs, None))
        current_state = current_state.apply_action(action)

    if debug:
        from go import go_board
        print(f"Game Over: {current_state.get_result()}")
        print(current_state.board)
        print(go_board.get_territory_numba(current_state.board))
    reward = current_state.get_result()  # 결과

    final_data = []
    for i, (st, ap, _) in enumerate(data):
        adjusted_result = reward if i % 2 == 0 else -reward
        final_data.append((st, ap, adjusted_result))
    print(f"Game Result: {reward}")
    return final_data

#####################################
# 5. 리플레이 버퍼
#####################################
class ReplayBuffer:
    def __init__(self, capacity=10000, device=None):
        """
        (state_tensor, action_probs, result)를 저장.
        state_tensor: (17, H, W)
        action_probs: (H*W + 1,)  # pass 포함
        result: scalar (1, -1, 0)
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def store(self, game_data):
        """game_data: List of (state_tensor, action_probs, result)"""
        self.buffer.extend(game_data)

    def sample(self, batch_size):
        """
        배치 샘플링 후, (states, action_probs, results) 텐서 반환.
        states: (B, 17, H, W)
        action_probs: (B, H*W + 1)
        results: (B,)
        """
        if len(self.buffer) < batch_size:
            return None, None, None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, action_probs, results = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        action_probs = torch.tensor(np.array(action_probs), dtype=torch.float32, device=self.device)
        results = torch.tensor(np.array(results), dtype=torch.float32, device=self.device)
        return states, action_probs, results

#####################################
# 6. 모델 학습 함수
#####################################
@timeit
def train_model(model: AlphaGoZeroNet, replay_buffer: ReplayBuffer, batch_size, epochs, learning_rate=1e-3, earlystopping=20, optimizer_state_dict=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.optimizer = optimizer
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    model.train()
    model.losses[len(model.losses)+1] = []
    best_model = None

    min_epoch = len(replay_buffer.buffer) // batch_size
    min_epoch *= (min_epoch) ** 0.5

    for epoch in range(epochs):
        states, action_probs, results = replay_buffer.sample(batch_size)
        if states is None:
            print("Not enough samples in replay buffer.")
            return False

        policy_output, value_output = model(states.to(model.device))

        # Policy Loss: 크로스 엔트로피 (예측과 타깃 비교)
        policy_loss = -torch.sum(action_probs * torch.log(policy_output + 1e-7), dim=1).mean()
        value_output = value_output.squeeze()
        value_loss = F.mse_loss(value_output, results)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.losses[len(model.losses)].append(loss.item())

        if earlystopping is not None:
            if best_model is None or loss.item() == min(model.losses[len(model.losses)]):
                best_model = model.copy()
                patience = earlystopping
            elif epoch > min_epoch:
                patience -= 1
                if patience == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Early Stopping")
                    break

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
    model = best_model

#####################################
# 7. 실제 학습 루프
#####################################
def self_play_worker(args):
    """멀티프로세싱용 단일 게임 처리 함수"""
    try:
        model, init_state, num_simulations, temperature = args
        return self_play(model, init_state, num_simulations=num_simulations, temperature=temperature)
    except KeyboardInterrupt:
        return None

def train(board_size=19, num_iterations=10, games_per_iteration=3, num_simulations=50, batch_size=16, epochs=100, learning_rate=1e-3, earlystopping=20, capacity=2000, device=None, pretrained_model_path=None, save_model_path=None, num_workers=4):
    model = AlphaGoZeroNet(board_size=board_size)
    optimizer_state_dict = None
    if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
        model.load(pretrained_model_path)
    device = torch.device(device)
    print(device)
    model = model.to(device)
    model.verbose = False
    model.to(torch.float32)
    
    replay_buffer = ReplayBuffer(capacity=capacity, device=device)

    try:
        for it in range(num_iterations):
            print(f"\n=== Iteration {it+1} / {num_iterations} ===")
            
            # print(f"Starting self-play with {games_per_iteration} games...")
            # Multi Process
            # empty_board = np.zeros((board_size, board_size), dtype=np.int32)
            # states = [(model, State(empty_board.copy(), current_player=1), num_simulations, 1.0) for _ in range(games_per_iteration)]
            # with mp.Pool(processes=num_workers) as pool:
            #     results = pool.map(self_play_worker, states)
            # for game_data in results:
            #     if game_data is not None:
            #         replay_buffer.store(game_data)
            
            # Single Process
            for g in range(games_per_iteration):
                print(f"Starting self-play with {g} / {games_per_iteration} game...")
                # 초기 바둑판 상태(모두 빈칸), 흑 선공
                empty_board = np.zeros((board_size, board_size), dtype=np.int32)
                init_state = State(empty_board, current_player=1, previous_board=None)

                # 한 판 자가 대국
                game_data = self_play(
                    model,
                    init_state, 
                    num_simulations=num_simulations, 
                    temperature=1.0
                )
                replay_buffer.store(game_data)
            
            print("Training...")
            train_model(model, replay_buffer, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, earlystopping=earlystopping, optimizer_state_dict=optimizer_state_dict)
            if save_model_path is None:
                save_model_path = f'models/cho_pha_go'
            model.save(save_model_path)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    # with 구문 사용 시 별도 Pool 종료 불필요

if __name__ == "__main__":
    board_size = 5  # 테스트용 작은 보드
    model = AlphaGoZeroNet(board_size=board_size)
    model.load('models/cho_pha_go_5x5.pt')
    model.verbose = False
    self_play(model, State(np.zeros((board_size, board_size), dtype=np.int32), current_player=1, previous_board=None), num_simulations=800, temperature=1.0, debug=True)
