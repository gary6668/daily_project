# 实现2048的棋盘，移动逻辑，合并，随机生成新格子
import numpy as np
import random

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.reset()

    def reset(self):
        """重置棋盘并放入两个随机方块"""
        self.board.fill(0)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        """在空白位置随机生成一个 2 或 4"""
        empty_positions = list(zip(*np.where(self.board == 0)))
        if not empty_positions:
            return
        r, c = random.choice(empty_positions)
        self.board[r, c] = 4 if random.random() < 0.1 else 2

    def move(self, direction):
        """进行一次移动 direction: 0=up, 1=right, 2=down, 3=left"""
        old_board = self.board.copy()  # 保存旧棋盘
        k = (direction + 1) % 4        # 将方向转为左移
        rot = np.rot90(self.board, k)  # 旋转棋盘
        moved, gained = self._move_left(rot)
        self.board = np.rot90(moved, -k)  # 转回去
        self.score += gained

        if not np.array_equal(self.board, old_board):
            self.add_random_tile()

    def _move_left(self, board):
        """执行一次左移和合并"""
        new_board = np.zeros_like(board)
        gained = 0

        for r in range(4):
            row = board[r][board[r] != 0]  # 去掉0，压紧
            new_row = []
            i = 0
            while i < len(row):
                if i + 1 < len(row) and row[i] == row[i + 1]:
                    merged = row[i] * 2
                    new_row.append(merged)
                    gained += merged
                    i += 2
                else:
                    new_row.append(row[i])
                    i += 1
            new_board[r, :len(new_row)] = new_row
        return new_board, gained
