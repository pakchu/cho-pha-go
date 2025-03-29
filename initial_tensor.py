from cho_pha_go_train import AlphaGoZeroNet, MCTSNode
from minigo.minigo import Position
from minigo.features import to_default_tensor
import numpy as np

board_size = 5  # 테스트용 작은 보드
model = AlphaGoZeroNet(board_size=board_size)
model.load('models/cho_pha_go_5x5.pt', 'cpu')

p1 = Position(np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]), to_play=1)

p2 = Position(np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]), to_play=-1)

for p in [p1, p2]:
    pol, exp_val = model(to_default_tensor(p).to(model.device))
    pol = pol.cpu().detach().numpy()
    pass_prob = pol.flatten()[-1]
    probs = pol.flatten()[:-1].reshape(board_size, board_size)
    print(probs)
    print(pass_prob)
    print(exp_val)