import numpy as np
import pygame
import torch
import warnings
from go.go_board import FastState as State
from cho_pha_go_train import AlphaGoZeroNet, ReplayBuffer, train_model

warnings.filterwarnings("ignore")

class InteractiveGo:
    def __init__(self, 
                 board_size=5, 
                 cell_size=60, 
                 margin=40, 
                 bottom_panel_height=60,
                 window_title="Go Game"):
        """
        Args:
            board_size (int): 바둑판 크기 (예: 5, 9, 19)
            cell_size (int): 한 칸의 픽셀 크기
            margin (int): 테두리 여백
            bottom_panel_height (int): 하단 버튼 영역 높이
            window_title (str): 윈도우 제목
        """
        pygame.init()
        
        # 인스턴스 변수 설정
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = margin
        self.bottom_panel_height = bottom_panel_height

        # 화면 크기 계산
        self.screen_width = self.board_size * self.cell_size + 2 * self.margin
        self.screen_height = self.board_size * self.cell_size + 2 * self.margin + self.bottom_panel_height

        # Pygame 창 초기화
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(window_title)

        # 색상 정의
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BROWN = (181, 101, 29)
        self.GRAY = (150, 150, 150)
        self.DARK_GRAY = (80, 80, 80)

    def draw_board(self):
        """바둑판 배경 + 선 그리기"""
        self.screen.fill(self.BROWN)

        # 바둑판 선 그리기
        for i in range(self.board_size):
            # 가로선
            start_x = self.margin
            start_y = self.margin + i * self.cell_size
            end_x = self.margin + (self.board_size - 1) * self.cell_size
            end_y = start_y
            pygame.draw.line(self.screen, self.BLACK, (start_x, start_y), (end_x, end_y), 1)

            # 세로선
            start_x = self.margin + i * self.cell_size
            start_y = self.margin
            end_x = start_x
            end_y = self.margin + (self.board_size - 1) * self.cell_size
            pygame.draw.line(self.screen, self.BLACK, (start_x, start_y), (end_x, end_y), 1)

    def draw_stone(self, x, y, color):
        """(x, y) 위치에 color 색 돌 그리기"""
        center_x = self.margin + x * self.cell_size
        center_y = self.margin + y * self.cell_size
        radius = self.cell_size // 2 - 2
        pygame.draw.circle(self.screen, color, (center_x, center_y), radius)

    def draw_button(self, text, x, y, width, height, color):
        """단순 직사각형 버튼 + 텍스트"""
        pygame.draw.rect(self.screen, color, (x, y, width, height))
        font = pygame.font.Font(None, 24)
        text_surface = font.render(text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(text_surface, text_rect)
        

    def is_button_pressed(self, pos, x, y, width, height):
        """pos가 버튼 범위(x, y, width, height) 내인지 체크"""
        return (x <= pos[0] <= x + width) and (y <= pos[1] <= y + height)

    def update_display(self, state: State):
        """전체 화면 업데이트 (보드 + 돌 + 버튼)"""
        self.draw_board()
        board = state.board
        for yy in range(self.board_size):
            for xx in range(self.board_size):
                if board[yy, xx] == 1:
                    self.draw_stone(xx, yy, self.BLACK)
                elif board[yy, xx] == -1:
                    self.draw_stone(xx, yy, self.WHITE)

        # reset 버튼
        self.draw_button("Reset", 10, self.screen_height - 50, 100, 40, self.GRAY)
        # skip 버튼
        self.draw_button("Skip", 120, self.screen_height - 50, 100, 40, self.GRAY)
        # undo 버튼 
        self.draw_button("Undo", 230, self.screen_height - 50, 100, 40, self.GRAY)
        pygame.display.flip()

    # ---------------------------
    # 사람 vs 사람
    # ---------------------------
    def run_player_vs_player(self):
        """
        사람 vs 사람 (오프라인) 플레이.
        """
        empty_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        game_state = State(board=empty_board, current_player=1)
        running = True

        self.update_display(game_state)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()

                    # 1) Reset 버튼 클릭 체크
                    if self.is_button_pressed(mouse_pos, 10, self.screen_height - 50, 100, 40):
                        # Reset
                        empty_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
                        game_state = State(empty_board, current_player=1)
                        self.update_display(game_state)
                        continue

                    # 2) 바둑판 클릭 처리
                    board_x = (mouse_pos[0] - self.margin) // self.cell_size
                    board_y = (mouse_pos[1] - self.margin) // self.cell_size

                    # 범위 확인
                    if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                        if game_state.is_valid_move(board_x, board_y):
                            game_state = game_state.apply_action((board_x, board_y))
                            # 게임 종료 체크
                            if game_state.is_terminal():
                                winner = game_state.get_result()
                                if winner == 1:
                                    print("흑(1) 승리!")
                                elif winner == -1:
                                    print("백(-1) 승리!")
                                else:
                                    print("무승부!")
                            self.update_display(game_state)

            pygame.display.flip()

        pygame.quit()

        
    def run_player_vs_ai(
        self, 
        player_black=True, 
        model_path=None, 
        device="cpu",
    ):
        """
        사람 vs AI 모드 + (옵션) 대국 데이터 학습에 활용.
        사람이 두는 수 / AI가 두는 수를 모두 기록.
        """

        # 1) 모델 로드
        if model_path is None:
            model_path = f"models/cho_pha_go"
        else:
            model_path = f"{model_path}"
        if model_path.endswith(f'_{self.board_size}x{self.board_size}.pt'):
            pass
        else:
            model_path += f'_{self.board_size}x{self.board_size}.pt'
        agent = AlphaGoZeroNet(board_size=self.board_size)
        agent.to(device)
        agent.load(model_path)
        agent.eval()
        agent.verbose = True
        replay_buffer = ReplayBuffer(capacity=10000, device=device)

        # 2) 상태 초기화
        empty_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        game_state = State(board=empty_board, current_player=1, memorize_before=True)

        # 플레이어 순서 결정 (turn=1: 사람, -1:AI)
        if player_black:
            turn = 1  # 사람 = 흑
        else:
            turn = -1  # 사람 = 백

        running = True
        self.update_display(game_state)

        # -----------------------------
        # 저장용: (state_tensor, action_probs, None) 리스트
        # 게임 종료 후 최종 결과(승/패/무승부)를 붙여서 replay_buffer에 넣음
        # -----------------------------
        data_this_game = []

        while running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_pos = pygame.mouse.get_pos()

                        # 1) Reset 버튼 체크
                        if self.is_button_pressed(mouse_pos, 10, self.screen_height - 50, 100, 40):
                            empty_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
                            game_state = State(empty_board, current_player=1, memorize_before=True)
                            turn = 1 if player_black else -1
                            self.update_display(game_state)
                            # Reset 시엔 이전 수 기록을 지워야 하나?
                            data_this_game.clear()
                            continue
                        # Skip 버튼 체크
                        elif self.is_button_pressed(mouse_pos, 120, self.screen_height - 50, 100, 40):
                            if turn == 1:
                                game_state.pass_count += 1
                                turn = -turn
                                game_state.current_player = -game_state.current_player
                                if game_state.is_terminal():
                                    winner = game_state.get_result()

                                    if winner == 1:
                                        print(f"흑(1) {'플레이어' if player_black else '조파고'} 승리!")
                                    elif winner == -1:
                                        print(f"백(-1) {'조파고' if player_black else '플레이어'} 승리!")
                                    else:
                                        print("무승부!")
                                    final_data = []
                                    
                                    for (st, ap, _) in data_this_game:
                                        final_data.append((st, ap, winner))
                                        
                                    if replay_buffer is not None:
                                        replay_buffer.store(final_data)
                                        
                                    data_this_game.clear()
                                    empty_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
                                    game_state = State(empty_board, current_player=1, memorize_before=True)
                                    turn = 1 if player_black else -1
                                    self.update_display(game_state)

                        # Undo 버튼 체크
                        elif self.is_button_pressed(mouse_pos, 230, self.screen_height - 50, 100, 40):
                            if len(data_this_game) > 0:
                                data_this_game.pop()
                                game_state = game_state.last
                                turn = -turn
                                self.update_display(game_state)
                                continue
                            else:
                                print("이전 수가 없습니다.")
                                continue
                                    
                        # 2) 바둑판 클릭 좌표
                        board_x = (mouse_pos[0] - self.margin) // self.cell_size
                        board_y = (mouse_pos[1] - self.margin) // self.cell_size

                        # (A) 사람이 둘 차례
                        if turn == 1:
                            if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                                if game_state.is_valid_move(board_x, board_y):
                                    # 1) 현재 상태 텐서
                                    state_tensor = game_state.to_tensor()

                                    # 2) 사람이 둔 수 → one-hot
                                    board_sz = self.board_size
                                    human_probs = np.zeros((board_sz * board_sz,), dtype=np.float32)
                                    idx = board_y * board_sz + board_x
                                    human_probs[idx] = 1.0

                                    # 3) 임시로 (state, one-hot, None) 저장
                                    data_this_game.append((state_tensor.squeeze(0), human_probs, None))

                                    # 착수
                                    game_state = game_state.apply_action((board_x, board_y))
                                    turn = -turn
                                    self.update_display(game_state)

                                    # 종료 체크
                                    if game_state.is_terminal():
                                        winner = game_state.get_result()
                                        # 데이터에 결과를 부여
                                        final_data = []
                                        for (st, ap, _) in data_this_game:
                                            final_data.append((st, ap, winner))
                                        # 버퍼에 저장
                                        if replay_buffer is not None:
                                            replay_buffer.store(final_data)
                                        data_this_game.clear()
                                else:
                                    print("유효하지 않은 수입니다.")
                        
                        # (B) AI가 둘 차례
                        else:
                            
                            agent.eval()
                            with torch.no_grad():
                                state_tensor = game_state.to_tensor().unsqueeze(0)
                                tensor, policy_np = agent(state_tensor)

                            move = agent.make_move(game_state)
                            
                            # when ai skips & game is valid
                            if move is not None and not game_state.is_valid_move(*move):
                                running = False
                                break

                            # (state_tensor, policy 전체, None) 저장
                            data_this_game.append((state_tensor.squeeze(0).squeeze(0), policy_np, None))

                            # 착수

                            game_state = game_state.apply_action(move)
                            turn = -turn
                            self.update_display(game_state)

                            if game_state.is_terminal():
                                winner = game_state.get_result()
                                # 전체 데이터에 결과 부여
                                final_data = []
                                for (st, ap, _) in data_this_game:
                                    final_data.append((st, ap, winner))
                                if replay_buffer is not None:
                                    replay_buffer.store(final_data)
                                data_this_game.clear()
                            
            except KeyboardInterrupt:
                running = False
                break
            
        train_model(
            model=agent,
            replay_buffer=replay_buffer,
            batch_size=1,
            epochs=10,
        )
        agent.save(
            '/'.join(model_path.split('/')[:-1] + ['vs_human_trained_' + model_path.split('/')[-1]]) if 'vs_human' not in model_path else model_path
        )
        
        pygame.display.flip()

        pygame.quit()


    def run_ai_vs_ai(self, model_path=None, device='cpu'):
        """
        AI vs AI 대국.
        """
        # 1) 모델 로드
        if model_path is None:
            model_path = f"models/cho_pha_go"
        else:
            model_path = f"{model_path}"
        if model_path.endswith(f'_{self.board_size}x{self.board_size}.pt'):
            pass
        else:
            model_path += f'_{self.board_size}x{self.board_size}.pt'
        agent_black = AlphaGoZeroNet(board_size=self.board_size)
        agent_white = AlphaGoZeroNet(board_size=self.board_size)
        agent_black.to(device)
        agent_white.to(device)
        agent_black.load(model_path)
        agent_white.load(model_path)
        agent_black.eval()
        agent_white.eval()
        agent_black.verbose = False
        agent_white.verbose = False

        empty_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        game_state = State(board=empty_board, current_player=1)

        running = True
        self.update_display(game_state)

        while running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # 2) AI 차례
                        agent = agent_black if game_state.current_player == 1 else agent_white
                        with torch.no_grad():
                            state_tensor = game_state.to_tensor().unsqueeze(0)
                            tensor, policy_np = agent(state_tensor)

                        move = agent.make_move(game_state)
                        if move is not None and not game_state.is_valid_move(*move):
                            running = False
                            break

                        game_state = game_state.apply_action(move)
                        self.update_display(game_state)

                        if game_state.is_terminal():
                            winner = game_state.get_result()
                            if winner == 1:
                                print("흑(1) 승리!")
                            elif winner == -1:
                                print("백(-1) 승리!")
                            else:
                                print("무승부!")

            except KeyboardInterrupt:
                running = False
                break

        pygame.display.flip()
        pygame.quit()
        

if __name__ == "__main__":
    # 예시 실행
    # 1) 9x9 보드에서 사람 vs 사람
    # game = InteractiveGo(board_size=9)
    # game.run_player_vs_player()

    # 2) 5x5 보드에서 사람(흑) vs AI
    game = InteractiveGo(board_size=5)
    game.run_ai_vs_ai()
