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
    def __init__(self, board_size=19):
        super(AlphaGoZeroNet, self).__init__()
        self.device = None
        self.board_size = board_size
        self.losses = defaultdict(list)
        self.optimizer = None
        # 네트워크 크기 동적 조정
        if board_size <= 9:
            self.num_filters = 64  # 작은 보드에서는 필터 수를 줄임
            self.num_residual_blocks = 3  # Residual Block 개수 축소
        elif board_size <= 13:
            self.num_filters = 128
            self.num_residual_blocks = 8
        else:
            self.num_filters = 256  # 기본 크기
            self.num_residual_blocks = 19

        # 첫 Conv
        self.conv1 = nn.Conv2d(17, self.num_filters, kernel_size=3, padding=1)

        # Residual Block 정의
        self.residual_blocks = nn.ModuleList(
            [self._build_residual_block() for _ in range(self.num_residual_blocks)]
        )

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size * 2, board_size * board_size),
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
        """
        Residual Block 정의
        """
        return nn.Sequential(
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_filters)
        )

    def forward(self, x):
        """
        모델 순전파
        """
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
        # 파라미터 확인
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        # 버퍼 확인
        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device
        # 기본값
        return torch.device("cpu")
    
    def save(
        self, 
        path, 
        # optimizer
    ): 
        tail = f'{self.board_size}x{self.board_size}' in path
        path = path + f'_{self.board_size}x{self.board_size}.pt' if not tail else path
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                # 'optimizer_state_dict': self.optimizer.state_dict(),
                'losses': self.losses
            }, 
            path
        )
        print(f"Model saved to {path}.")
    
    def load(
        self,
        path,
        # optimizer
    ):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint['losses']
        print(f"Model loaded from {path}.")

    def mcts_search(self, state: State, num_simulations=800):
        """
        MCTS 탐색 루틴. root 노드로부터 시뮬레이션 여러 번 수행.
        """
        root = MCTSNode(state)

        for _ in range(num_simulations):
            node = root

            # 1) Selection
            #    자식 노드가 존재하고, 상태가 터미널이 아닐 동안 select_child
            while node.child_nodes and not node.state.is_terminal():
                node = node.select_child()

            # 2) Expansion
            if not node.state.is_terminal():
                with torch.no_grad():
                    # 네트워크 추론
                    state_tensor = node.state.to_tensor().to(self.device)
                    policy, value = self(state_tensor.unsqueeze(0))  # (1, board_size^2), (1, 1)
                    policy = policy.squeeze(0).cpu().numpy()  # shape: (board_size^2,)
                    value = value.item()

                board_size = state.board.shape[0]
                # policy를 보드 형태로 reshape
                policy_2d = policy.reshape(board_size, board_size)

                # 합법 착수 가져오기
                legal_moves = node.state.get_legal_actions()
                action_probs = []
                total_prob = 0.0

                for move in legal_moves:
                    if move is None:
                        # 패스 처리
                        pass_prob = policy.mean()  # 또는 임의의 소값
                        action_probs.append((None, pass_prob))
                        total_prob += pass_prob
                    else:
                        x, y = move
                        p = policy_2d[y][x]
                        action_probs.append(((x, y), p))
                        total_prob += p

                # 합이 0이면(아주 드문 경우) 균등 분포로 대체
                if total_prob > 0:
                    action_probs = [(a, p / total_prob) for (a, p) in action_probs]
                else:
                    action_probs = [(a, 1.0 / len(legal_moves)) for a in legal_moves]

                # 자식 노드 확장
                node.expand(action_probs)
            else:
                # 터미널이면 value를 그대로 사용
                value = node.state.get_result()

            # 3) Backpropagation
            #    현재 node부터 부모 방향으로 value를 반전시키며 업데이트
            while node is not None:
                node.update(value)
                value = -value
                node = node.parent

        return root
    
    def make_move(self, state: State, temperature=1.0):
        """
        현재 상태에서 MCTS를 이용해 행동을 선택
        """
        self.eval()
        root = self.mcts_search(state, num_simulations=800)
        return root.best_action(temperature=temperature)



#####################################
# 2. Markov Chain Tree Search(MCTS)
#####################################
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

        # 변경 1) children -> 3개 자료구조
        #  - child_actions: list of actions
        #  - child_nodes:   list of MCTSNode
        #  - children_map:  dict[action -> index in the above lists]
        self.child_actions = []
        self.child_nodes = []
        self.children_map = {}

        self.visits = 0
        self.value_sum = 0
        self.policy = 0  # 이 노드(action)에 대한 policy 확률

    def expand(self, action_probs):
        """
        자식 노드 생성 + 정책(사전확률) 할당
        action_probs: [(action, prob), ...]
        """
        for (action, prob) in action_probs:
            # action이 이미 있으면 pass
            if action in self.children_map:
                continue

            # 새로운 자식 노드 생성
            child_state = self.state.apply_action(action)
            child_node = MCTSNode(child_state, parent=self)
            child_node.policy = prob

            # 리스트/딕셔너리에 등록
            idx = len(self.child_actions)
            self.child_actions.append(action)
            self.child_nodes.append(child_node)
            self.children_map[action] = idx

    def select_child(self, exploration_param=1.4):
        """
        UCB를 최대화하는 자식 노드 선택 (벡터화 + 리스트 방식)
        """
        if not self.child_nodes:
            return None

        # 방문 횟수가 parent가 0이면 log(0) 때문에 문제 -> 작은 epsilon
        log_parent_visits = np.log(self.visits + 1e-8)

        # 자식 노드의 정보 벡터화
        node_count = len(self.child_nodes)
        visits_arr = np.empty(node_count, dtype=np.float64)
        value_sum_arr = np.empty(node_count, dtype=np.float64)
        policy_arr = np.empty(node_count, dtype=np.float64)

        for i, child in enumerate(self.child_nodes):
            visits_arr[i] = child.visits
            value_sum_arr[i] = child.value_sum
            policy_arr[i] = child.policy

        # avg_value = value_sum / visits  (visits=0 이면 inf로 설정)
        avg_value = np.zeros_like(value_sum_arr, dtype=np.float64)

        mask_visited = (visits_arr > 0)
        avg_value[mask_visited] = value_sum_arr[mask_visited] / visits_arr[mask_visited]

        # UCB = avg_value + c_puct * policy * sqrt( log(parent.visits) / visits )
        # 방문수=0인 곳은 inf
        with np.errstate(divide='ignore', invalid='ignore'):
            ucb = avg_value + exploration_param * policy_arr * np.sqrt(
                log_parent_visits / (visits_arr + 1e-8)
            )
            # 방문수가 0인 곳 -> inf로 설정
            ucb[~mask_visited] = np.inf

        best_idx = np.argmax(ucb)
        return self.child_nodes[best_idx]  # 해당 자식 노드 반환

    def update(self, value):
        """
        백업(역전파)
        """
        self.visits += 1
        self.value_sum += value

    def best_action(self, temperature=1.0):
        """
        최적 행동 반환. (자식 중 하나를 골라 해당 action 반환)
        temperature=0이면 방문 수가 가장 많은 행동,
        그 외엔 방문 횟수 비례해서 샘플링
        """
        if not self.child_nodes:
            return None

        node_count = len(self.child_nodes)
        visits_arr = np.array([child.visits for child in self.child_nodes], dtype=np.float64)

        if temperature == 0:
            # 방문수가 가장 많은 액션
            best_idx = np.argmax(visits_arr)
            return self.child_actions[best_idx]
        else:
            # 방문 수를 기준으로 확률 분포 생성
            visits_arr = visits_arr ** (1.0 / temperature)
            sum_v = np.sum(visits_arr)

            if sum_v < 1e-12:
                # 모두 0 이면 균등분포
                probs = np.ones(node_count) / node_count
            else:
                probs = visits_arr / sum_v

            chosen_idx = np.random.choice(node_count, p=probs)
            return self.child_actions[chosen_idx]





#####################################
# 4. 자가 대국(self-play)
#####################################
@timeit
def self_play(model: AlphaGoZeroNet, init_state: State, num_simulations=800, temperature=1.0):
    """
    한 판을 완주할 때까지 MCTS로 행동을 선택하고,
    (state_tensor, action_probs, 최종결과)를 리턴.
    """
    data = []  # (state_tensor, action_probs, result)
    current_state = init_state

    while not current_state.is_terminal():
        # 1) 현재 상태에서 MCTS 탐색
        #    (model.mcts_search가 내부에서 MCTSNode 구조를 사용하여 root.child_nodes/child_actions를 채운다고 가정)
        root = model.mcts_search(current_state, num_simulations)

        # 2) 방문 횟수 기반 행동 확률 계산
        #    - root.child_nodes : list of MCTSNode
        #    - root.child_actions : list of actions
        #    따라서 방문 횟수도 리스트/배열로 뽑아서 확률화 가능
        if not root.child_nodes:
            # 자식이 없는 경우(종료 상태) -> 바로 탈출
            break

        visits = np.array([child.visits for child in root.child_nodes], dtype=np.float64)
        actions = root.child_actions  # 길이가 child_nodes와 동일

        # 정규화
        probs = visits / np.sum(visits)

        # 3) temperature 적용하여 행동 선택
        if temperature == 0:
            # 방문수가 가장 많은 액션 선택
            best_idx = np.argmax(visits)
            action = actions[best_idx]
        else:
            # 방문 수를 temperature^(1/temperature) power로 변형 후 확률화
            visits = visits ** (1.0 / temperature)
            sum_v = np.sum(visits)
            if sum_v < 1e-12:
                # 전체가 0 -> 균등분포
                probs = np.ones_like(visits) / len(visits)
            else:
                probs = visits / sum_v
            # 샘플링
            chosen_idx = np.random.choice(len(actions), p=probs)
            action = actions[chosen_idx]

        # 4) 학습 데이터 저장
        #    - 모델 입력: 현재 state's to_tensor()
        #    - action_probs: 보드 전체( board_size * board_size )에 대한 분포
        #      root 노드의 방문 횟수 분포를 이용해 저장
        board_size = current_state.board.shape[0]
        full_action_probs = np.zeros((board_size * board_size,), dtype=np.float32)
        for i, act in enumerate(actions):
            if act is None:
                # 패스인 경우는 보드 전체에서 인덱스가 없음
                # (패스를 하나의 확률로 다루려면 별도 처리해야 함)
                continue
            x, y = act
            idx = y * board_size + x
            full_action_probs[idx] = probs[i]

        state_tensor = current_state.to_tensor()  # shape: (C, H, W) or (17, board_size, board_size)
        data.append((state_tensor, full_action_probs, None))

        # 5) 실제 게임 상태 업데이트
        #    - action이 None이면 패스
        current_state = current_state.apply_action(action)

    # 게임 종료 후 승패 결과
    result = current_state.get_result()  # 1(흑승) / -1(백승) / 0(무승부)

    # 데이터에 결과 추가
    final_data = []
    for (st, ap, _) in data:
        final_data.append((st, ap, result))

    return final_data



#####################################
# 5. 리플레이 버퍼
#####################################
class ReplayBuffer:
    def __init__(self, capacity=10000, device = None):
        """
        자가 대국에서 얻은 (state_tensor, action_probs, result)를 저장하는 버퍼
        state_tensor: (17, H, W)
        action_probs: (H*W,) (또는 board_size^2,) 
        result: 스칼라(흑승=1, 백승=-1, 무승부=0)
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device
    def store(self, game_data):
        """
        game_data: List of (state_tensor, action_probs, result)
        - state_tensor.shape == (17, H, W)  (numpy 또는 torch.tensor)
        - action_probs.shape == (H*W,)
        - result: scalar
        """
        self.buffer.extend(game_data)

    def sample(self, batch_size):
        """
        배치 크기(batch_size)만큼 랜덤 샘플링하여, 
        (states, action_probs, results)를 텐서 형태로 반환

        states.shape == (B, 17, H, W)
        action_probs.shape == (B, H*W)
        results.shape == (B,)
        """
        if len(self.buffer) < batch_size:
            return None, None, None

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]  # list of (state, probs, result)

        # 각각 분리
        states, action_probs, results = zip(*batch)

        # 파이썬 tuple -> numpy array -> torch tensor
        # states는 각각 (17, H, W)이므로, np.array(states) -> (B, 17, H, W)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)

        action_probs = torch.tensor(np.array(action_probs), dtype=torch.float32, device=self.device)
        results = torch.tensor(np.array(results), dtype=torch.float32, device=self.device)

        return states, action_probs, results



#####################################
# 6. 모델 학습 함수
#####################################
@timeit
def _train(model: AlphaGoZeroNet, replay_buffer, optimizer, batch_size, epoch, epochs):
    states, action_probs, results = replay_buffer.sample(batch_size)
    if states is None:
        print("Not enough samples in replay buffer.")
        return False

    # 순전파
    policy_output, value_output = model(states.to(model.device))

    # (1) Policy Loss: 크로스 엔트로피 (또는 음의 로그 우도)
    #    action_probs(타겟)와 policy_output(예측) 간 비교
    policy_loss = -torch.sum(action_probs * torch.log(policy_output + 1e-7), dim=1).mean()

    # (2) Value Loss: MSE
    value_output = value_output.squeeze()  # shape: (B,)
    value_loss = F.mse_loss(value_output, results)

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.losses[len(model.losses)].append(loss.item())
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
    return True

@timeit
def train_model(model: AlphaGoZeroNet, replay_buffer: ReplayBuffer, batch_size, epochs, learning_rate=1e-3, optimizer_state_dict=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.optimizer = optimizer
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    model.train()
    # iteration 마다 loss 들을 저장하는 리스트
    model.losses[len(model.losses)+1] = []
    for epoch in range(epochs):
        if not _train(model, replay_buffer, optimizer, batch_size, epoch, epochs):
            break
        
    # return optimizer

def play_self_game(model, num_simulations=800, board_size=19):
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
    return game_data

#####################################
# 7. 실제 학습 루프
#####################################
def train(
    board_size=19, 
    num_iterations=10,     # 자가 대국 + 학습을 몇 번 반복할지
    games_per_iteration=2, # 매 iteration마다 몇 판의 자가 대국을 할지
    num_simulations=50,    # MCTS 시뮬레이션 횟수
    batch_size=16,
    epochs=100,
    learning_rate=1e-3,
    capacity=2000,
    device=None, 
    pretrained_model_path=None,
    save_model_path=None,
    multi_process_num=4
):
    model = AlphaGoZeroNet(board_size=board_size)
    optimizer_state_dict = None
    if pretrained_model_path is not None:
        if os.path.exists(pretrained_model_path):
            model.load(pretrained_model_path)
    device = torch.device(device)
    model = model.to(device)
    model.to(torch.float32)
    
    # 리플레이 버퍼
    replay_buffer = ReplayBuffer(capacity=capacity, device=device)

    try:
        for it in range(num_iterations):
            print(f"\n=== Iteration {it+1} / {num_iterations} ===")

            # 1) 자가 대국 진행 후 데이터 수집
            model.to('cpu')
            processes = deque([mp.Process(target=play_self_game, args=(model, num_simulations, board_size)) for _ in range(games_per_iteration)])
            
            for i in range(len(processes)):
                if i< multi_process_num:
                    processes[i].start()
                else:
                    break
            for i in range(len(processes)):
                if i< multi_process_num:
                    processes[i].join()
                else:
                    break
            for i in range(len(processes)):
                if i< multi_process_num:
                    game_data = play_self_game(model, num_simulations, board_size)
                    replay_buffer.store(game_data)
                else:
                    break
            for i in range(len(processes)):
                if i < multi_process_num:
                    processes.popleft()
                
            replay_buffer.store(game_data)
            
            # 2) 버퍼에서 샘플을 뽑아 모델 학습
            model.to(device)
            print("Training...")
            train_model(
                model, 
                replay_buffer, 
                batch_size=batch_size, 
                epochs=epochs, 
                learning_rate=learning_rate,
                optimizer_state_dict=optimizer_state_dict
            )
    
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    # except Exception as e:
    #     print(e)
        

    if save_model_path is None:
        save_model_path = f'models/cho_pha_go'
    model.to('cpu')
    model.save(save_model_path)


if __name__ == "__main__":
    board_size = 5  # 테스트용 작은 크기 예시
    train(board_size=board_size)
    breakpoint()
