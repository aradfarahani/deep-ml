import numpy as np
from typing import Optional

class MCTSNode:
    def __init__(self, state: int, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0  # wins credited to the player who moved INTO this node

    def is_leaf(self):
        return len(self.children) == 0

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = c * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration

def mcts_search(initial_state: int, max_value: int, iterations: int, seed: int = 42) -> int:
    np.random.seed(seed)
    root = MCTSNode(initial_state)

    def node_depth(n):
        d = 0
        while n.parent is not None:
            d += 1
            n = n.parent
        return d

    def rollout(state, player):
        # `player` = whose turn it is at `state`; returns the winning player's id (0/1)
        if state >= max_value:
            return (1 - player) if state == max_value else player
        cur = player
        while True:
            state += np.random.choice([1, 2])
            mover = cur
            cur = 1 - cur
            if state == max_value:
                return mover        # landed exactly -> mover wins
            if state > max_value:
                return 1 - mover    # overshot      -> mover loses

    for _ in range(iterations):
        node = root
        # Selection
        while not node.is_leaf() and node.state < max_value:
            node = max(node.children.values(), key=lambda n: n.ucb1())
        # Expansion
        if node.state < max_value and node.visits > 0:
            for action in (1, 2):
                ns = node.state + action
                if ns <= max_value + 1:
                    node.children[action] = MCTSNode(ns, parent=node)
            node = list(node.children.values())[0]
        # Simulation
        player_to_move = node_depth(node) % 2
        winner = rollout(node.state, player_to_move)
        # Backpropagation
        credit = 1 - player_to_move  # player who moved into `node`
        n = node
        while n is not None:
            n.visits += 1
            if credit == winner:
                n.value += 1.0
            credit = 1 - credit
            n = n.parent

    if not root.children:
        return 1
    return max(root.children.items(), key=lambda x: x[1].visits)[0]