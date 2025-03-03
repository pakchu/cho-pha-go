import numba, torch, numpy as np

# @numba.njit
def get_neighbors_numba(x, y, w, h):
    """
    (x, y)의 상하좌우 좌표를 반환. 
    """
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < w - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < h - 1:
        neighbors.append((x, y + 1))
    return neighbors

# @numba.njit
def get_group_numba(board, start_x, start_y):
    """
    (start_x, start_y)에 있는 돌( board[start_y,start_x] )과 
    연결된 그룹(동일 색, 상하좌우 연결)을 BFS/스택 방식으로 추적.
    return: list of (x, y)
    """
    color = board[start_y, start_x]
    stack = [(start_x, start_y)]
    visited = []
    visited_set = set()  # numba가 set을 완벽 지원X -> 대안은 boolean array or dict

    while len(stack) > 0:
        x, y = stack.pop()
        if (x, y) in visited_set:
            continue
        visited_set.add((x, y))
        visited.append((x, y))

        # 이웃 탐색
        neighs = get_neighbors_numba(x, y, board.shape[1], board.shape[0])
        for (nx, ny) in neighs:
            if board[ny, nx] == color and (nx, ny) not in visited_set:
                stack.append((nx, ny))

    return visited

# @numba.njit
def get_liberties_numba(board, group):
    """
    그룹(group)에 대해 공배(자유도) 좌표 세트를 구함.
    group: list of (x, y)
    return: list of (x, y) 공배
    """
    liberties = []
    lib_set = set()  # numba에서 set 사용은 제한적 -> 대안으로 dict나 list 중복체크
    for (gx, gy) in group:
        neighs = get_neighbors_numba(gx, gy, board.shape[1], board.shape[0])
        for (nx, ny) in neighs:
            if board[ny, nx] == 0:
                if (nx, ny) not in lib_set:
                    lib_set.add((nx, ny))
                    liberties.append((nx, ny))
    return liberties

# @numba.njit
def remove_dead_stones_numba(board, x, y, current_player):
    """
    (x, y)에 착수 후, 주변 상대 돌 중 공배가 0인 그룹 제거.
    board: 2D numpy array (흑=1, 백=-1, 빈=0)
    current_player: 1 or -1
    return: (new_board, dead_count)  -> 제거된 돌 수
    """
    # board는 copy된 것을 전제로 수정
    h, w = board.shape
    opponent = -current_player
    dead_count = 0

    # (x, y) 주변 좌표
    neighs = get_neighbors_numba(x, y, w, h)
    for (nx, ny) in neighs:
        if board[ny, nx] == opponent:
            group = get_group_numba(board, nx, ny)
            libs = get_liberties_numba(board, group)
            if len(libs) == 0:
                # 캡처
                for (gx, gy) in group:
                    board[gy, gx] = 0
                dead_count += len(group)
    return board, dead_count

# @numba.njit
def is_valid_move_numba(board, prev_board, x, y, current_player):
    """
    착수 가능 여부 확인 (자충수/패규칙 등)
    board: 현재 board (수 놓기 전)
    prev_board: 직전 착수 직후 보드
    """
    if prev_board is None:
        prev_board = np.zeros_like(board)
        
    h, w = board.shape
    # 1) 범위 체크
    if not (0 <= x < w and 0 <= y < h):
        return False
    # 2) 이미 돌이 있는 곳
    if board[y, x] != 0:
        return False

    # 2-1) 완전한 집인 경우: (x, y)의 상하좌우 칸이 모두 current_player의 돌이고,
    #     그 돌들이 하나의 연결 그룹을 이루면, 굳이 돌을 둘 필요가 없음.
    neighs = get_neighbors_numba(x, y, w, h)
    if neighs and all(board[ny, nx] == current_player for (nx, ny) in neighs):
        group = get_group_numba(board, neighs[0][0], neighs[0][1])
        if all((nx, ny) in group for (nx, ny) in neighs):
            return False
    
    # 3) 임시로 돌을 놓음
    temp_board = board.copy()
    temp_board[y, x] = current_player

    
    all_neighbors_empty = True
    unique_corner_condition = False
    if (x, y) in [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]:
        for (nx, ny) in neighs:
            if board[ny, nx] == 0:
                if all(board[neigbor_neibor] != 0 for neigbor_neibor in get_neighbors_numba(nx, ny, w, h)):
                    unique_corner_condition = True
                    break
            else:
                all_neighbors_empty = False
                
        if not unique_corner_condition and all_neighbors_empty:
            return False
                
    
    # 4) remove_dead_stones
    temp_board, dead_count = remove_dead_stones_numba(temp_board, x, y, current_player)

    # 5) 착수한 돌의 그룹 + 공배 체크
    group = get_group_numba(temp_board, x, y)
    libs = get_liberties_numba(temp_board, group)
    if len(libs) == 0:
        # 자충수
        return False

    # 6) 패 규칙: temp_board == prev_board
    #    (numba에서 array_equal을 직접 구현 or np.array_equal 대체)
    if prev_board is not None and prev_board.shape == board.shape:
        return ~np.all(temp_board == prev_board)
    else:
        raise ValueError("Invalid prev_board shape")

# @numba.njit
def get_territory_numba(board):
    """
    board 상의 빈 칸(0)들에 대해 Flood Fill 방식으로
    각각의 영역 크기와 인접 색 정보를 조사하여
    흑/백 집 개수를 최종적으로 반환한다.

    - board: 2D numpy array (흑=1, 백=-1, 빈=0)
    return: (black_territory, white_territory)
    """
    h, w = board.shape
    visited = np.zeros((h, w), dtype=np.int32)

    black_territory = 0
    white_territory = 0

    for y in range(h):
        for x in range(w):
            if board[y, x] == 0 and visited[y, x] == 0:
                # 아직 방문하지 않은 빈 칸 -> 하나의 영역을 BFS/DFS로 찾음
                q = [(x, y)]
                visited[y, x] = 1
                empty_coords = []
                neighbor_colors = set()

                while len(q) > 0:
                    cx, cy = q.pop()
                    empty_coords.append((cx, cy))

                    # 상하좌우 확인
                    neighbors = get_neighbors_numba(cx, cy, w, h)
                    for nx, ny in neighbors:
                        if board[ny, nx] == 0:
                            # 빈 칸이면 같은 영역
                            if visited[ny, nx] == 0:
                                visited[ny, nx] = 1
                                q.append((nx, ny))
                        else:
                            # 돌(흑=1, 백=-1)이면 해당 색을 기록
                            # board[ny, nx] != 0인 경우만
                            neighbor_colors.add(board[ny, nx])
                
                # 영역이 인접한 돌 색깔들 neighbor_colors
                # 만약 {1}만 있다면 -> 흑만 인접 -> 흑 집
                # 만약 {-1}만 있다면 -> 백만 인접 -> 백 집
                # 둘 다 있거나 없는 경우(이론상 없지만) -> 공배
                if len(neighbor_colors) == 1:
                    color = neighbor_colors.pop()
                    if color == 1:
                        black_territory += len(empty_coords)
                    elif color == -1:
                        white_territory += len(empty_coords)
                    # pop() 했으므로 다시 넣을 필요는 없음
                # 공배(또는 복잡 케이스) -> 무시
                # (실전 바둑에선 '양쪽 모두 인접'이라도 부분적인 생사관계 등에 따라
                #  세부 해석이 다르지만, 여기서는 단순 무효로 처리)
    return black_territory, white_territory

# @numba.njit
def check_independent_liberties(board, liberties):
    """
    주어진 공배(liberties)가 독립된 영역인지 확인.
    - Flood Fill로 공배의 독립된 영역을 계산.
    """
    h, w = board.shape
    visited = np.zeros((h, w), dtype=np.int32)
    independent_areas = 0

    for lx, ly in liberties:
        if visited[ly, lx] == 0:
            # 새로운 독립된 공배 영역 발견 -> Flood Fill
            queue = [(lx, ly)]
            visited[ly, lx] = 1

            while queue:
                cx, cy = queue.pop()
                for nx, ny in get_neighbors_numba(cx, cy, w, h):
                    if (nx, ny) in liberties and visited[ny, nx] == 0:
                        visited[ny, nx] = 1
                        queue.append((nx, ny))

            independent_areas += 1

    return independent_areas


# @numba.njit
def calculate_territory_with_alive_groups(board):
    """
    살아있는 돌에 의해 형성된 정당한 집 계산.
    """
    h, w = board.shape
    visited = np.zeros((h, w), dtype=np.int32)

    black_territory = 0
    white_territory = 0

    for y in range(h):
        for x in range(w):
            if board[y, x] == 0 and visited[y, x] == 0:
                # 빈 칸 탐색
                queue = [(x, y)]
                visited[y, x] = 1
                empty_coords = []
                neighbor_colors = set()

                while queue:
                    cx, cy = queue.pop()
                    empty_coords.append((cx, cy))

                    for nx, ny in get_neighbors_numba(cx, cy, w, h):
                        if board[ny, nx] == 0:
                            if visited[ny, nx] == 0:
                                visited[ny, nx] = 1
                                queue.append((nx, ny))
                        else:
                            neighbor_colors.add(board[ny, nx])

                # 집 판별
                if len(neighbor_colors) == 1:
                    color = neighbor_colors.pop()
                    if color == 1:  # 흑돌
                        black_territory += len(empty_coords)
                    elif color == -1:  # 백돌
                        white_territory += len(empty_coords)

    return black_territory, white_territory         

# @numba.njit
def check_alive_groups(board):
    """
    살아있는 돌 그룹의 집만 인정.
    """
    h, w = board.shape
    visited = np.zeros((h, w), dtype=np.int32)
    black_territory = 0
    white_territory = 0

    for y in range(h):
        for x in range(w):
            if board[y, x] != 0 and visited[y, x] == 0:
                # 돌 그룹 탐색
                group = get_group_numba(board, x, y)
                for gx, gy in group:
                    visited[gy, gx] = 1

                # 공배 계산
                liberties = get_liberties_numba(board, group)
                if len(liberties) < 2:
                    continue  # 공배가 두 개 미만이면 집 형성 불가

                # 독립된 공배 영역 확인
                independent_liberties = check_independent_liberties(board, liberties)
                if independent_liberties >= 2:
                    # 살아있는 그룹 판별
                    color = board[y, x]
                    if color == 1:  # 흑돌
                        black_territory += len(liberties)
                    elif color == -1:  # 백돌
                        white_territory += len(liberties)

    return black_territory, white_territory


class FastState:
    def __init__(
        self,
        board: np.ndarray,
        current_player: int,
        dum = 0.5,
        previous_board: np.ndarray = None,
        pass_count: int = 0,
        player_1_dead_stones: int = 0,
        player_minus_1_dead_stones: int = 0,
        history: list = None,
        memorize_before: bool = False,
        last = None,
        least_number_of_stones = 0.3,
    ):
        self.board = board
        self.current_player = current_player
        self.dum = dum
        self.previous_board = previous_board
        self.pass_count = pass_count
        self.player_1_dead_stones = player_1_dead_stones
        self.player_minus_1_dead_stones = player_minus_1_dead_stones
        self.memorize_before = memorize_before
        self.last = last
        self.least_number_of_stones = least_number_of_stones
        self.valid_moves = []
        if history is None:
            self.history = []
        else:
            self.history = history

    def copy(self):
        return FastState(
            board=self.board.copy(),
            current_player=self.current_player,
            previous_board=self.previous_board.copy() if self.previous_board is not None else None,
            pass_count=self.pass_count,
            player_1_dead_stones=self.player_1_dead_stones,
            player_minus_1_dead_stones=self.player_minus_1_dead_stones,
            history=self.history[:],  # shallow copy of list
            memorize_before=self.memorize_before,
            last = self.last.copy() if self.last is not None else None,
        )
        
    def is_pass_available(self):
        return len(self.get_legal_actions()) <= 2

    def is_valid_move(self, x, y):
        # Numba 함수 호출
        prev_board = self.previous_board if self.previous_board is not None else None
        return is_valid_move_numba(self.board, prev_board, x, y, self.current_player)

    def apply_action(self, action):
        if action is None:
            # 패스
            new_state = self.copy()
            new_state.previous_board = self.board.copy()
            new_state.current_player = -self.current_player
            new_state.pass_count += 1
            return new_state

        (x, y) = action
        if not self.is_valid_move(x, y):
            raise ValueError(f"Invalid move at ({x}, {y})")

        new_board = self.board.copy()
        new_board[y, x] = self.current_player

        new_state = FastState(
            board=new_board,
            current_player=self.current_player,
            previous_board=self.board.copy(),
            pass_count=0,
            player_1_dead_stones=self.player_1_dead_stones,
            player_minus_1_dead_stones=self.player_minus_1_dead_stones,
            history=self.history[:],
            memorize_before=self.memorize_before,
        )
        # 착수 후 상대 돌 제거
        # -> numba 함수로 호출
        nb_board, dead_count = remove_dead_stones_numba(new_board, x, y, self.current_player)
        new_state.board = nb_board
        if -self.current_player == 1:
            new_state.player_1_dead_stones += dead_count
        else:
            new_state.player_minus_1_dead_stones += dead_count

        # 다음 플레이어
        new_state.current_player = -self.current_player
        new_state.history.append((x, y, self.current_player))
        if self.memorize_before:
            new_state.last = self.copy()
        return new_state

    def get_legal_actions(self):
        if self.valid_moves:
            return self.valid_moves
        actions = []
        h, w = self.board.shape
        prev_board = self.previous_board if self.previous_board is not None else None
        for yy in range(h):
            for xx in range(w):
                if is_valid_move_numba(self.board, prev_board, xx, yy, self.current_player):
                    actions.append((xx, yy))
        # 패스
        actions.append(None)
        self.valid_moves = actions
        return actions

    def is_terminal(self):
        if self.pass_count >= 2:
            return True
        return False

    def get_result(self):
        # 깔린 수가 적은데 승부를 냈다면 무승부
        # if (self.board==0).sum() < self.board.ravel().size * self.least_number_of_stones:
        #     return 0
        
        black_territory, white_territory = check_alive_groups(self.board)
        black_score = black_territory + self.player_minus_1_dead_stones
        white_score = white_territory + self.player_1_dead_stones
        diff = black_score - self.dum - white_score
        # print(diff)
        return 1 if diff > 0 else -1 if diff < 0 else 0
    
    def get_verbose_result(self):
        # 깔린 수가 적은데 승부를 냈다면 무승부
        # if (self.board==0).sum() < self.board.ravel().size * self.least_number_of_stones:
        #     return 0
        
        black_territory, white_territory = check_alive_groups(self.board)
        black_score = black_territory + self.player_minus_1_dead_stones
        white_score = white_territory + self.player_1_dead_stones
        diff = black_score - self.dum - white_score
        # print(diff)
        return 1 if diff > 0 else -1 if diff < 0 else 0, black_score, white_score

    def to_tensor(self) -> torch.Tensor:
        """
        현재 게임 상태를 AlphaGo Zero 논문의 입력 형식인 (1, 17, H, W) 형태의 텐서로 변환.
        
        구성:
        - 채널 0~7: 지난 8 시점의 보드 상태에서, 현재 플레이어의 돌 위치 (1이면 돌, 0이면 없음)
        - 채널 8~15: 지난 8 시점의 보드 상태에서, 상대 플레이어의 돌 위치 (1이면 돌, 0이면 없음)
        - 채널 16: 현재 플레이어 표시 (흑이면 모든 위치 1, 백이면 모든 위치 0)
        
        보드 상태는 게임 시작부터 현재까지 self.history의 (x, y, player) 정보를
        순차적으로 적용해 재구성합니다.
        """
        h, w = self.board.shape  # 예: 19, 19
        tensor = np.zeros((17, h, w), dtype=np.float32)
        
        # (1) 게임 시작부터 현재까지의 보드 상태 재구성
        # board_states[i]는 i번째 move 후의 보드 상태 (board_states[0]는 빈 보드)
        board_states = []
        current_state = np.zeros((h, w), dtype=np.int8)
        board_states.append(current_state.copy())
        for move in self.history:
            x, y, player = move
            # (캡처 등은 고려하지 않고 단순히 돌을 놓는다고 가정)
            current_state[y, x] = player
            board_states.append(current_state.copy())
        
        # 현재까지 진행된 move 수 (board_states의 마지막 인덱스 = len(self.history))
        t = len(self.history)
        
        # (2) 지난 8 시점의 보드 상태로부터 채널 채우기
        # i=0 : t-7번째 상태, i=7 : t번째 (현재) 상태
        for i in range(8):
            state_index = t - 7 + i  # t가 7 미만이면 음수가 될 수 있음.
            if state_index < 0:
                # 게임 시작 이전: 빈 보드 상태
                state = board_states[0]
            else:
                state = board_states[state_index]
            # 현재 플레이어의 돌: 해당 위치가 self.current_player인 경우 1, 아니면 0
            tensor[i] = (state == self.current_player).astype(np.float32)
            # 상대 플레이어의 돌: 해당 위치가 -self.current_player인 경우 1, 아니면 0
            tensor[i + 8] = (state == -self.current_player).astype(np.float32)
        
        # (3) 채널 16: 현재 플레이어 표시
        # 논문에서는 흑(1) 차례이면 1, 백(-1) 차례이면 0으로 표기
        tensor[16] = 1.0 if self.current_player == 1 else 0.0
        
        # 최종적으로 (17, H, W)를 (1, 17, H, W) 텐서로 변환하여 반환
        return torch.tensor(tensor, dtype=torch.float32)#.unsqueeze(0)

    
def test_is_valid_move_complete_territory():
    # 5x5 보드 생성 (초기에는 모두 빈칸 0)
    board = np.zeros((5, 5), dtype=np.int32)
    current_player = 1  # 예: 흑돌
    
    # 3x3 블록 내에서 (2,2)만 빈 칸으로 두고 나머지는 모두 흑돌로 채웁니다.
    # 좌표 (1,1)부터 (3,3)까지 채우되 (2,2)는 건너뜁니다.
    for y in range(1, 4):
        for x in range(1, 4):
            if (x, y) != (2, 2):
                board[y, x] = current_player
                
    board[4, 1] = 1
    board[4, 3] = 1
    board[3, 4] = 1

    print(board)
    # 이전 보드는 모두 빈 상태로 초기화
    prev_board = np.zeros_like(board)
    
    # (2,2)는 빈 칸이지만 상하좌우 (즉, (2,1), (2,3), (1,2), (3,2))가 모두 흑돌이고,
    # 이들 돌이 하나의 연결 그룹을 이루므로, 이미 완벽한 집으로 간주되어 착수할 필요가 없습니다.
    valid = is_valid_move_numba(board, prev_board, 2, 2, current_player)

    print("Test Complete Territory: Move at (2,2) should be invalid (False). Got:", valid)
    assert valid == False, "Expected move at (2,2) to be invalid (already complete territory)."

    # (2, 4) 는 빈 칸이지만 상하좌우 (즉, (2,3), (1,4), (3,4))가 모두 흑돌이고,
    # 이들 돌이 하나의 연결 그룹을 이루므로, 이미 완벽한 집으로 간주되어 착수할 필요가 없습니다.
    valid = is_valid_move_numba(board, prev_board, 2, 4, current_player)
    print("Test Complete Territory: Move at (2,4) should be invalid (False). Got:", valid)
    assert valid == False, "Expected move at (2,4) to be invalid (already complete territory)."
    
    # (4, 4)는 빈 칸이지만 상하좌우 (즉, (4,3), (3,4))가 모두 흑돌이고,
    # 이들 돌이 하나의 연결 그룹을 이루므로, 이미 완벽한 집으로 간주되어 착수할 필요가 없습니다.
    valid = is_valid_move_numba(board, prev_board, 4, 4, current_player)
    print("Test Complete Territory: Move at (4,4) should be invalid (False). Got:", valid)
    assert valid == False, "Expected move at (4,4) to be invalid (already complete territory)."
    
    # (0,0)은 주변에 돌이 없으므로 정상적인 착수로 간주되어야 합니다.
    valid_edge = is_valid_move_numba(board, prev_board, 0, 0, current_player)
    print("Test Non-surrounded: Move at (0,0) should be valid (True). Got:", valid_edge)
    assert valid_edge == True, "Expected move at (0,0) to be valid."

    print("All tests passed.")

if __name__ == "__main__":
    test_is_valid_move_complete_territory()