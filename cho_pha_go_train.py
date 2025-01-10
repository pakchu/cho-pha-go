import copy
import os
import multiprocessing as mp
import numpy as np
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from go_tf import Position
from features import position_to_tensor
import warnings
from utils import timeit, time_record, time_records

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
#####################################
# 1. 알파고 제로 신경망 정의
#####################################
class AlphaGoZeroNet(nn.Module):
    @time_record
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
            self.num_residual_blocks = 3  # Residual Block 개수 축소
        elif board_size <= 13:
            self.num_filters = 128
            self.num_residual_blocks = 8
        else:
            self.num_filters = 256  # 기본 크기
            self.num_residual_blocks = 19

        # 첫 Conv
        self.conv1 = nn.Conv2d(21, self.num_filters, kernel_size=3, padding=1)

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

    @time_record
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

    @time_record
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

        
    @time_record
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = self.get_device()
        return self
    
    @time_record
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

    @time_record
    def copy(self):
        """
        모델 복사
        """
        model = AlphaGoZeroNet(board_size=self.board_size)
        model.load_state_dict(self.state_dict())
        model.device = self.device
        model.losses = copy.copy(self.losses)
        return model
    
    @time_record
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
    
    @time_record
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

    @torch.no_grad()
    @time_record
    def mcts_search(self, state: Position, num_simulations=800):
        """
        MCTS 탐색 루틴. root 노드로부터 시뮬레이션 여러 번 수행.
        """
        root = MCTSNode(state)

        for _ in range(num_simulations):
            node = root
            # 1) Selection
            while len(node.children) > 0 and not node.state.is_game_over():
                node = node.select_child()

            # 2) Expansion
            if not node.state.is_game_over():
                with torch.no_grad():
                    # 네트워크 추론
                    state_tensor = position_to_tensor(node.state).to(self.device)
                    policy, value = self(state_tensor.unsqueeze(0))
                    policy = policy.squeeze().cpu().numpy()  # shape: (board_size^2,)
                    value = value.item()

                board_size = state.board.shape[0]
                # policy를 보드 형태로 reshape
                # 바둑판 좌표 확률
                policy_2d = policy.reshape(board_size, board_size)  # 첫 n^2 요소만 reshape

                # 합법 착수
                legal_moves = node.state.all_legal_move_coords()
                action_probs = []
                total_prob = 0.0
                for move in legal_moves:
                    # if move is None:
                    #     # 패스 처리
                    #     action_probs.append((None, pass_prob))
                    #     total_prob += pass_prob
                    # else:
                    x, y = move
                    p = policy_2d[y][x]
                    action_probs.append(((x, y), p))
                    total_prob += p

                # 합이 0이면(정말 드문 케이스) 균등 분포로 대체
                if total_prob > 0:
                    action_probs = [(a, p / total_prob) for a, p in action_probs]
                else:
                    action_probs = [(a, 1.0 / len(legal_moves)) for a in legal_moves]

                # 자식 노드 확장
                node.expand(action_probs)
            else:
                # 터미널이면 value를 그대로 사용
                if node.state.is_game_over():
                    value = node.state.result()

            # 3) Backpropagation
            # 현재 node부터 부모 방향으로 value를 반전시키며 업데이트
            while node is not None:
                node.update(value)
                value = -int(value)
                node = node.parent

        return root
    
    @time_record
    def make_move(self, state: Position, temperature=1.0):
        """
        현재 상태에서 MCTS를 이용해 행동을 선택
        """
        self.eval()
        root = self.mcts_search(state, num_simulations=800)
        return root.best_action(temperature=temperature, verbose=self.verbose)


#####################################
# 2. MCTS 구현
#####################################
class MCTSNode:
    @time_record
    def __init__(self, state: Position, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.policy = 0  # 이 노드(action)에 대한 policy 확률

    @time_record
    def expand(self, action_probs):
        """
        현재 노드에서 가능한 모든 합법 착수에 대해 자식 노드를 만들고,
        해당 자식 노드에 policy(사전확률)을 할당
        """
        for action, prob in action_probs:
            if action not in self.children:
                # if action is None:
                #     self.children[action] = MCTSNode(
                #         self.state.pass_move(),
                #         parent=self
                #     )
                if self.state.is_move_legal(action):
                    self.children[action] = MCTSNode(
                        self.state.play_move(action, color=self.state.to_play),
                        parent=self
                    )
                    self.children[action].policy = prob

    @time_record
    def select_child(self, exploration_param=1.4):
        """UCB를 최대화하는 자식 노드 선택"""
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            score = self._ucb_score(child, exploration_param)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    @time_record
    def _ucb_score(self, child, exploration_param):
        if child.visits == 0:
            return float('inf')  # 아직 방문 안 한 자식은 우선적으로 탐색
        avg_value = child.value_sum / child.visits
        # parent.visits가 0인 경우는 거의 없으므로 생략
        ucb = avg_value + exploration_param * child.policy * \
              np.sqrt(np.log(self.visits) / child.visits)
        return ucb

    @time_record
    def update(self, value):
        """백업(역전파)"""
        self.visits += 1
        self.value_sum += value

    @time_record
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
            visits = np.array([child.visits for child in self.children.values()])
            visits = visits ** (1.0 / temperature)
            total = np.sum(visits)
            probs = visits / total
            probs = probs.reshape(-1)
            actions = list(self.children.keys())
            if verbose:
                print(f"Visits: {visits}")
                print(f"Probs: {probs}")
                print(f"Actions: {actions}")
            return actions[np.random.choice(np.arange(len(actions)), p=probs.tolist())]






#####################################
# 4. 자가 대국(self-play)
#####################################
@timeit
@time_record
def self_play(model: AlphaGoZeroNet, init_state: Position, num_simulations=800, temperature=1.0, debug=False, device='cpu'):
    """
    한 판을 완주할 때까지 MCTS로 행동을 선택하고,
    (state_tensor, action_probs, 최종결과)를 리턴.
    """
    model = model.to(device)
    data = []  # (state_tensor, action_probs, result)
    current_state = init_state

    while not current_state.is_game_over():
        # 1) 현재 상태에서 MCTS 진행
        root = model.mcts_search(current_state, num_simulations)
        # 2) 방문 횟수 기반 행동 확률 계산
        visits = np.array([child.visits for child in root.children.values()])
        actions = list(root.children.keys())
        probs = visits / np.sum(visits)

        # 3) temperature 적용하여 행동 선택
        if temperature == 0:
            # 방문수가 가장 많은 액션 선택
            action = max(root.children, key=lambda a: root.children[a].visits)
            a = 1
        else:
            # 방문 수를 temperature^(-1) power로 변형 후 확률화
            visits = visits ** (1.0 / temperature)
            visits /= np.sum(visits)
            if len(actions) > 0:

                action = actions[np.random.choice(len(actions), p=visits)]

            else:
                if current_state.is_game_over():
                    break

        # 4) 학습 데이터 저장
        #    - 모델 입력: 현재 state's to_tensor()
        #    - action_probs: 합법 착수 개수만큼이 아닌, 보드 전체(19*19)에 대응
        #      => 여기서는 root 노드의 policy 출력을 그대로 방문 횟수 분포로 사용.
        #      이 또한 (board_size*board_size,) 형태
        board_size = current_state.board.shape[0]
        full_action_probs = np.zeros((board_size * board_size,), dtype=np.float32)
        for i, act in enumerate(actions):
            # if act is None:
            #     full_action_probs[-1] = probs[i]
            #     continue
            x, y = act
            idx = y * board_size + x
            full_action_probs[idx] = probs[i]

        state_tensor = position_to_tensor(current_state).to(device)
        data.append((state_tensor, full_action_probs, None))

        # 5) 실제 게임 상태 업데이트
        # if action is None:
        #     current_state = current_state.pass_move()
        # else:
        try:
            current_state = current_state.play_move(action, color=current_state.to_play)
        except:
            breakpoint()

    # 게임 종료 후 승패 결과
    result = current_state.result()

    # 결과를 모두 업데이트
    final_data = []
    if result != 0:
        for (st, ap, _) in data:
            final_data.append((st, ap, result))
    else:
        for i, (st, ap, _) in enumerate(data):
            if i % 2 == 0:
                final_data.append((st, ap, 1))
            else:
                final_data.append((st, ap, -1))

    return final_data


#####################################
# 5. 리플레이 버퍼
#####################################
class ReplayBuffer:
    @time_record
    def __init__(self, capacity=10000, device = None):
        """
        자가 대국에서 얻은 (state_tensor, action_probs, result)를 저장하는 버퍼
        state_tensor: (17, H, W)
        action_probs: (H*W,) (또는 board_size^2,) 
        result: 스칼라(흑승=1, 백승=-1, 무승부=0)
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device
    @time_record
    def store(self, game_data):
        """
        game_data: List of (state_tensor, action_probs, result)
        - state_tensor.shape == (17, H, W)  (numpy 또는 torch.tensor)
        - action_probs.shape == (H*W,)
        - result: scalar
        """
        self.buffer.extend(game_data)

    @time_record
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

        states_tensor = torch.concat([state.unsqueeze(0) for state in states], dim=0)
        action_probs = torch.tensor(np.array(action_probs), dtype=torch.float32, device=self.device)
        results = torch.tensor(np.array(results), dtype=torch.float32, device=self.device)

        return states_tensor, action_probs, results



#####################################
# 6. 모델 학습 함수
#####################################
@timeit
@time_record
def train_model(
    model: AlphaGoZeroNet, 
    replay_buffer: ReplayBuffer, 
    batch_size, 
    epochs, 
    learning_rate=1e-3, 
    earlystopping=20, 
    optimizer_state_dict=None
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.optimizer = optimizer
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    model.train()
    model.to()
    # iteration 마다 loss 들을 저장하는 리스트
    model.losses[len(model.losses)+1] = []
    best_model = None
    
    min_epoch = len(replay_buffer.buffer) // batch_size
    min_epoch *= (min_epoch) ** 0.5
    
    for epoch in range(epochs):
        
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
        
        # Early Stopping
        # print(earlystopping)

        if earlystopping is not None:
            if best_model is None or loss.item() == min(model.losses[len(model.losses)]):
                best_model = model.copy()
                patience = earlystopping
                # print('best model updated')
            elif epoch > min_epoch:
                patience -= 1
                if patience == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Early Stopping")
                    break

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
    model = best_model
    # return optimizer


#####################################
# 7. 실제 학습 루프
#####################################
@time_record
def self_play_worker(args):
    """멀티프로세싱에서 사용할 단일 게임 처리 함수"""
    # try:
    model, init_state, num_simulations, temperature = args
    return self_play(model, init_state, num_simulations=num_simulations, temperature=temperature)
    # except KeyboardInterrupt:
    #     return None

@time_record
def train(
    board_size=19, 
    num_iterations=10,     
    games_per_iteration=2, 
    num_simulations=50,    
    batch_size=16,
    epochs=100,
    learning_rate=1e-3,
    earlystopping=20,
    capacity=2000,
    device=None, 
    pretrained_model_path=None,
    save_model_path=None,
    num_workers=4  # 병렬 처리에 사용할 프로세스 수
):
    model = AlphaGoZeroNet(board_size=board_size)
    optimizer_state_dict = None
    if pretrained_model_path is not None:
        if os.path.exists(pretrained_model_path):
            model.load(pretrained_model_path)
    device = torch.device(device)
    model = model.to(device)
    model.verbose = False
    model.to(torch.float32)
    
    # 리플레이 버퍼
    replay_buffer = ReplayBuffer(capacity=capacity, device=device)

    try:
        for it in range(num_iterations):
            print(f"\n=== Iteration {it+1} / {num_iterations} ===")

            # 1) 자가 대국 진행 후 데이터 수집 (멀티프로세싱 활용)
            print(f"Starting self-play with {games_per_iteration} games...")
            # empty_board = np.zeros((board_size, board_size), dtype=np.int32)

            # # 병렬 처리를 위한 초기 상태 리스트
            # states = [
            #     (model, Position(board=empty_board.copy(), to_play=1), num_simulations, 1.0)
            #     for _ in range(games_per_iteration)
            # ]
            # Pool = mp.Pool(processes=num_workers)
            # with Pool as pool:
            #     results = pool.map(self_play_worker, states)

            # for game_data in results:
            #     if game_data is not None:
            #         replay_buffer.store(game_data)
            
            # multi processing을 사용하지 않고 단일 프로세스로 실행
            for _ in range(games_per_iteration):
                game_data = self_play(
                    model, 
                    Position(board=np.zeros((board_size, board_size), dtype=np.int32), to_play=1), 
                    num_simulations=num_simulations, 
                    temperature=1.0, 
                    debug=True, 
                    # device=device
                )
                model.to(device)
                replay_buffer.store(game_data)
            # model.to(device)
            # 2) 버퍼에서 샘플을 뽑아 모델 학습
            print("Training...")
            train_model(
                model, 
                replay_buffer, 
                batch_size=batch_size, 
                epochs=epochs, 
                learning_rate=learning_rate,
                earlystopping=earlystopping,
                optimizer_state_dict=optimizer_state_dict,
            )
    except KeyboardInterrupt:
        if save_model_path is None:
            save_model_path = f'models/cho_pha_go'
            model.save(save_model_path)
        import pandas as pd
        df = pd.Series(time_records).sort_values(ascending=False)
        pd.set_option('display.max_rows', None)
        print(df)
        print("KeyboardInterrupt")
    # finally:
    #     # 학습 중단 시 프로세스 종료
    #     # try:
    #     #     Pool.terminate()
    #     #     Pool.join()
    #     # except KeyboardInterrupt:
    #         pass
    


if __name__ == "__main__":
    board_size = 5  # 테스트용 작은 크기 예시
    pos = Position(board=np.zeros((board_size, board_size), dtype=np.int32), to_play=1)
    t = position_to_tensor(pos)
    breakpoint()
    model = AlphaGoZeroNet(board_size=board_size)
    model.load('models/cho_pha_go_5x5.pt')
    model.verbose = False
    self_play(model, Position(np.zeros((board_size, board_size), dtype=np.int32), to_play=1), num_simulations=800, temperature=1.0, debug=True)

