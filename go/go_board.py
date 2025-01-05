import numba, torch, numpy as np

@numba.njit
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

@numba.njit
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

@numba.njit
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

@numba.njit
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

@numba.njit
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

    # 3) 임시로 돌을 놓음
    temp_board = board.copy()
    temp_board[y, x] = current_player

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

@numba.njit
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

@numba.njit
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


@numba.njit
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

@numba.njit
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
        dum = 3.5,
        previous_board: np.ndarray = None,
        pass_count: int = 0,
        player_1_dead_stones: int = 0,
        player_minus_1_dead_stones: int = 0,
        history: list = None,
        memorize_before: bool = False,
        last = None,
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
        actions = []
        h, w = self.board.shape
        prev_board = self.previous_board if self.previous_board is not None else None
        for yy in range(h):
            for xx in range(w):
                if is_valid_move_numba(self.board, prev_board, xx, yy, self.current_player):
                    actions.append((xx, yy))
        # 패스
        actions.append(None)
        return actions

    def is_terminal(self):
        if self.pass_count >= 2:
            return True
        if len(self.get_legal_actions()) == 1:  # only None
            return True
        return False

    def get_result(self):
        black_territory, white_territory = check_alive_groups(self.board)
        black_score = black_territory + self.player_minus_1_dead_stones
        white_score = white_territory + self.player_1_dead_stones
        if black_score == white_score == 0:
            return 0  # 무승부
        return 1 if black_score - self.dum > white_score else -1
        

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

        # (2) 최근 6개 history 보드 채널 (channel 1~13)
        #     history[-1] = 가장 최근 상태
        #     history[-2] = 그 이전 ...
        max_moves = 12
        history_len = len(self.history)
        for i in range(max_moves):
            if i >= history_len:
                break
            move = self.history[- i - 1]
            x, y, player = move
            if player == 1:
                tensor[i//2 + 1, y, x] = 1.0
            else:
                tensor[7 + i//2, y, x] = -1.0
        
        # (3) channel 14~15는 사석 정보
        tensor[14, :, :] = self.player_1_dead_stones if self.current_player == 1 else self.player_minus_1_dead_stones
        tensor[15, :, :] = self.player_minus_1_dead_stones if self.current_player == 1 else self.player_1_dead_stones
        

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
