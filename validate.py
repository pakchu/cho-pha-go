from cho_pha_go_train import AlphaGoZeroNet, MCTSNode
from minigo.minigo import Position
from minigo.features import to_default_tensor
import numpy as np

board_size = 5  # 테스트용 작은 보드
model = AlphaGoZeroNet(board_size=board_size)
model.load('models/cho_pha_go_5x5.pt', 'cpu')

p = Position(np.array([
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 1, 0, 0, -1],
    [-1, -1, -1, -1, 0],
    [-1, -1, -1, -1, -1]
]), to_play=-1)

pol, exp_val = model(to_default_tensor(p).to(model.device))

root = MCTSNode(p, exploration=1.4)
root = root.search(model, p, num_simulations=200)


# actions = list(root.children.keys())
# ucb = [root._ucb_score(root.children[a]) for a in actions]
# chosen = max(list(zip(ucb, actions)))[1]
# print(root.children)

# if chosen is None or chosen == (3,4):
#     print(f'Test failed: chosen action is {chosen}')

# else:
#     print(f'Test passed: chosen action is {chosen}')
    
res = []
for _ in range(100):
    root = MCTSNode(p, exploration=0.5)
    root = root.search(model, p, num_simulations=200, network_trust=0.5)
    actions = list(root.children.keys())
    ucb = [root._ucb_score(root.children[a]) for a in actions]
    chosen = max(list(zip(ucb, actions)))[1]
    if chosen is None or chosen == (3,4):
        res.append(False)
    else:
        res.append(True)
    print(f'Pass rate: {sum(res)} / {len(res)}', end='\r')
print(f'\nPass rate: {sum(res)/len(res)}')