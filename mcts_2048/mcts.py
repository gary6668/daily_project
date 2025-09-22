import math
import random
import copy
import numpy as np
from game import Game2048

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.reward = 0.0

    def is_fully_expanded(self):
        return len(self.children) == 4

    def best_child(self, c_param=1.0):  # 🔧 调低探索系数，减少乱试
        best = -float('inf')
        best_kids = []
        N = self.visits
        lnN = math.log(N + 1.0)
        for action, child in self.children.items():
            Q = child.reward / (child.visits + 1e-9)
            U = c_param * math.sqrt(lnN / (child.visits + 1e-9))
            score = Q + U
            if score > best:
                best = score
                best_kids = [child]
            elif abs(score - best) < 1e-12:
                best_kids.append(child)
        return random.choice(best_kids)

def rollout(game, steps=30):  # 🔧 增加 rollout 步数
    """随机模拟 steps 步，奖励 = 分数 + 空位奖励 + 最大格奖励"""
    start_score = game.score
    for _ in range(steps):
        actions = [0, 1, 2, 3]
        random.shuffle(actions)
        moved = False
        for a in actions:
            old = game.board.copy()
            game.move(a)
            if not np.array_equal(old, game.board):
                moved = True
                break
        if not moved:
            break

    empty_cells = np.count_nonzero(game.board == 0)
    max_tile = np.max(game.board)
    return (game.score - start_score) + 0.1 * empty_cells + 0.01 * max_tile

def backpropagate(node, reward):
    while node:
        node.visits += 1
        node.reward += reward
        node = node.parent

def expand(node, game):
    actions = [0, 1, 2, 3]
    random.shuffle(actions)
    for a in actions:
        if a not in node.children:
            temp_game = copy.deepcopy(game)
            moved = temp_game.move(a)
            if moved:
                new_node = Node(temp_game.board, parent=node, action=a)
                node.children[a] = new_node
                return new_node, temp_game
    return node, game

def mcts_search(game, iterations=3000, c_param=1.0):  # 🔧 iterations加大
    root = Node(copy.deepcopy(game.board))

    for _ in range(iterations):
        node = root
        temp_game = Game2048()
        temp_game.board = game.board.copy()
        temp_game.score = game.score

        while node.is_fully_expanded() and node.children:
            node = node.best_child(c_param)
            temp_game.move(node.action)

        if not temp_game.game_over():
            node, temp_game = expand(node, temp_game)

        reward = rollout(temp_game, steps=30)
        backpropagate(node, reward)

    best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
    return best_action
