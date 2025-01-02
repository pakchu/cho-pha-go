import numpy as np
import torch


# TODO: 속도 개선을 위해 numpy array를 사용하고, 필요한 경우에만 PyTorch 텐서로 변환 또는 numba.jit 사용

class State:
    def __init__(
        self,
        board: np.ndarray,
        current_player: int,
        previous_board: np.ndarray = None,
        pass_count: int = 0,
        player_1_dead_stones: int = 0,
        player_minus_1_dead_stones: int = 0,
        history: list = None,
    ):
        """
        Args:
            board: 2D numpy array (흑:1, 백:-1, 빈칸:0)
            current_player: 1(흑) 혹은 -1(백)
            previous_board: 직전 착수 직후의 바둑판(패 규칙 확인 용)
            pass_count: 직전에 연속된 패스 횟수(0, 1, ...).
        """
        self.board = board
        self.current_player = current_player
        self.previous_board = previous_board
        self.pass_count = pass_count  # 연속 두 번 패스되면 종료로 간주
        self.player_1_dead_stones = player_1_dead_stones
        self.player_minus_1_dead_stones = player_minus_1_dead_stones
        if history is None:
            self.history = []
        else:
            self.history = history

    def copy(self):
        """딥카피 or 얕은 카피(필요에 맞게)."""
        return State(
            board=self.board.copy(),
            current_player=self.current_player,
            previous_board=self.previous_board.copy() if self.previous_board is not None else None,
            pass_count=self.pass_count
        )

    def get_neighbors(self, x, y):
        """(x, y)의 상하좌우 좌표"""
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < self.board.shape[1] - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < self.board.shape[0] - 1:
            neighbors.append((x, y + 1))
        return neighbors

    def get_group(self, x, y, visited=None):
        """
        (x, y)에 있는 돌과 연결된 그룹(동일 색, 상하좌우 연결).
        return: {(x1,y1), (x2,y2), ...} 형태의 set
        """
        
        # dfs로 구현
        if visited is None:
            visited = set()
        if (x, y) in visited:
            return visited
        visited.add((x, y))

        color = self.board[y][x]
        for (nx, ny) in self.get_neighbors(x, y):
            if self.board[ny][nx] == color:
                self.get_group(nx, ny, visited)
        return visited

    def get_liberties(self, group):
        """해당 그룹의 공배(자유도) 집합을 반환"""
        liberties = set()
        for (x, y) in group:
            for (nx, ny) in self.get_neighbors(x, y):
                if self.board[ny][nx] == 0:
                    liberties.add((nx, ny))
        return liberties

    def remove_dead_stones(self, x, y):
        """
        (x, y)에 착수 후 주변 상대 돌들 중 자유도가 0인 그룹을 제거.
        """
        opponent = -self.current_player
        neighbors = self.get_neighbors(x, y)
        for (nx, ny) in neighbors:
            if self.board[ny][nx] == opponent:
                group = self.get_group(nx, ny)
                libs = self.get_liberties(group)
                if len(libs) == 0:
                    # 캡처(제거)
                    for (gx, gy) in group:
                        self.board[gy][gx] = 0
                    if opponent == 1:
                        self.player_1_dead_stones += len(group)
                    else:
                        self.player_minus_1_dead_stones += len(group)

    def is_valid_move(self, x, y):
        """(x, y)에 착수 가능 여부 확인"""
        # 범위 밖 or 이미 돌이 있음
        if not (0 <= x < self.board.shape[1] and 0 <= y < self.board.shape[0]):
            return False
        if self.board[y][x] != 0:
            return False

        # 임시로 돌을 두어 본다.
        temp_board = self.board.copy()
        temp_board[y][x] = self.current_player

        # 착수 후 상대 돌 캡처, 돌 제거 전까진 current_player 유지
        test_state = State(temp_board, self.current_player, previous_board=self.board)
        test_state.remove_dead_stones(x, y)

        # 착수한 돌(그룹)의 자유도 확인
        group = test_state.get_group(x, y)
        libs = test_state.get_liberties(group)
        if len(libs) == 0:
            # 내 돌이 자충수 상태 -> 착수 불가(단, 착수와 동시에 상대 돌이 캡처되는 경우 아니면)
            # remove_dead_stones()가 이미 상대 돌은 제거했음
            return False

        # 패(ko) 규칙: 착수 후의 보드가 previous_board와 동일하면 안 됨
        if self.previous_board is not None and np.array_equal(test_state.board, self.previous_board):
            return False

        return True

    def apply_action(self, action):
        """
        action: (x, y) or None(패스)
        """
        # 패스 처리
        if action is None:
            # 다음 상태로 넘어감(pass_count+1)
            new_state = State(
                board=self.board.copy(),
                current_player=-self.current_player,
                previous_board=self.board.copy(),
                pass_count=self.pass_count + 1
            )
            return new_state

        (x, y) = action
        if not self.is_valid_move(x, y):
            raise ValueError(f"Invalid move at ({x}, {y})")

        new_board = self.board.copy()
        new_board[y][x] = self.current_player
        
        # 새 상태 생성, 상대 돌 제거 전까진 current_player 유지
        new_state = State(
            board=new_board,
            current_player=self.current_player,
            previous_board=self.board.copy(),
            pass_count=0,  # 착수하면 pass_count 리셋
            player_1_dead_stones=self.player_1_dead_stones,
            player_minus_1_dead_stones=self.player_minus_1_dead_stones,
            history=self.history.copy()
        )
        # 착수 후 상대 돌 제거
        new_state.remove_dead_stones(x, y)
        # 다음 플레이어로 변경
        new_state.current_player = -self.current_player
        new_state.history.append((x, y, self.current_player))
        return new_state

    def get_legal_actions(self):
        """모든 합법착수 + (옵션) 패스도 포함"""
        actions = []
        h, w = self.board.shape
        for yy in range(h):
            for xx in range(w):
                if self.is_valid_move(xx, yy):
                    actions.append((xx, yy))
        # 패스도 가능하다고 가정하면, actions.append(None) 추가
        actions.append(None)  # 패스
        return actions

    def is_terminal(self):
        """
        1) 연속 두 번 패스(pass_count >= 2)
        2) 또는 합법착수가 전혀 없는 경우(옵션)
        """
        if self.pass_count >= 2:
            return True

        # 착수할 곳이 아예 없으면(= 전부 invalid이거나 돌이 가득)
        # 굳이 pass를 기다리지 않고 종료 가능하도록(옵션)
        if len(self.get_legal_actions()) == 1:
            # get_legal_actions()에서 None(패스)만 있는 상황이라면
            return True
        return False

    def get_result(self):
        """돌 개수로 승패 비교(단순화)"""
        black_stones = np.sum(self.board == 1) - self.player_1_dead_stones
        white_stones = np.sum(self.board == -1) - self.player_minus_1_dead_stones
        if black_stones > white_stones:
            return 1
        elif white_stones > black_stones:
            return -1
        else:
            return 0
        
    def to_tensor(self) -> torch.Tensor:
        """
        총 17채널 (C=17)짜리 numpy array를 만든 뒤, (1, 17, H, W) 형태의 PyTorch 텐서로 반환.
        
        구체적 구성:
        - channel 0: 현재 보드 (흑=1, 백=-1, 빈=0)
        - channel 1~15: 최근 7, 8번의 history 보드 (가장 최근이 channel 1, 그 전이 channel 2, ...)

        - channel 16: 현재 플레이어 (흑 차례=1.0, 백 차례=0.0) 
        """
        h, w = self.board.shape
        c = 17
        tensor = np.zeros((c, h, w), dtype=np.float32)

        # (1) 채널 0 -> 현재 보드 (직접 흑=1, 백=-1, 빈=0 으로 저장)
        #     이미 self.board 자체가 {1, -1, 0} 값을 가지므로 그대로 복사
        tensor[0, :, :] = self.board

        # (2) 최근 8개 history 보드 채널 (channel 1~8)
        #     history[-1] = 가장 최근 상태
        #     history[-2] = 그 이전 ...
        max_moves = 15
        num_history = len(self.history)
        for i in range(max_moves):
            
            (x, y, c) = self.history[num_history - i - 1] if i < num_history else (0, 0, 0)
                
            tensor[i+1, y, x] = c
            

        # (3) channel 9~15는 사용 안 함 => 이미 0으로 초기화되어 있으므로 그대로 둠
        #     필요하면 여기에 다른 정보(패 정보, 사석 정보 등)를 넣을 수 있음.

        # (4) channel 16 -> 현재 플레이어 표시
        #     알파고 제로 논문에선 (흑 차례=1, 백 차례=0) 식의 binary plane
        #     (또는 흑=1, 백=-1)을 쓰는 구현도 있지만 여기선 1.0 / 0.0 예시
        if self.current_player == 1:
            # 흑 차례
            tensor[16, :, :] = 1.0
        else:
            # 백 차례
            tensor[16, :, :] = -1.0

        # 마지막으로 (17, H, W)를 (1, 17, H, W)로 만들어 리턴
        return torch.tensor(tensor, dtype=torch.float32)

            
if __name__ == "__main__":
    board = [[0, 0,  0,  0,  0],
            [0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0]]
    board = np.array(board)
    game_state = State(board=board, current_player=1)
    game_state=game_state.apply_action((0, 0))
    game_state=game_state.apply_action((1, 0))
    game_state=game_state.apply_action((1, 1))
    # breakpoint()
    game_state=game_state.apply_action((0, 1))
    breakpoint()