import random
from cho_pha_go_train import AlphaGoZeroNet, MCTSNode
from minigo.minigo import Position
from minigo.features import to_default_tensor
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Cho Pha Go')
parser.add_argument('--ai-black', default=False, action='store_true', help='AI plays black')
args = parser.parse_args()

board_size = 5  # 테스트용 작은 보드
model = AlphaGoZeroNet(board_size=board_size)
model.load('models/cho_pha_go_5x5.pt', 'cpu')

os.environ.setdefault('BOARD_SIZE', str(board_size))

p = Position()

ai_wins = []

ai_color = 1 if args.ai_black else -1

def ai_step(p: Position, model: AlphaGoZeroNet):
    action, _ = model.make_move(p, 0, 200, 0.5, 0.1)
    p = p.play_move(action)
    reset = False
    if p.is_game_over():
        ai_wins.append(p.result() == ai_color)
        p = Position()
        reset = True
    return p, reset

def random_step(p: Position):
    legal = p.all_legal_moves()
    legal_moves = [(y, x) for x in range(board_size) for y in range(board_size) if legal[y * board_size + x]] + [None]
    action = random.choice(legal_moves)
    p = p.play_move(action)
    reset = False
    if p.is_game_over():
        ai_wins.append(p.result() == ai_color)
        p = Position()
        reset = True
    return p, reset

for _ in range(100):
    r = False
    while not r:
        if args.ai_black:
            p, r = ai_step(p, model)
            if not r:
                p, r = random_step(p)
        else:
            p, r = random_step(p)
            if not r:
                p, r = ai_step(p, model)
    print(f'AI win rate: {sum(ai_wins)} / {len(ai_wins)}'.ljust(100), end='\r')

print(f'\nAI win rate: {sum(ai_wins) / len(ai_wins)}')