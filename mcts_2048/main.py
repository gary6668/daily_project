import time
import numpy as np
from game import Game2048
from mcts import mcts_search
from utils import print_board

if __name__ == "__main__":
    MAX_STEPS = 200
    ITERATIONS = 3000

    game = Game2048()
    print("初始棋盘:")
    print_board(game.board)

    for step in range(MAX_STEPS):
        if game.game_over():
            print(f"游戏结束！总得分: {game.score}")
            break

        start_time = time.time()
        action = mcts_search(game, iterations=ITERATIONS, c_param=1.0)
        end_time = time.time()

        game.move(action)

        # 统计信息
        max_tile = np.max(game.board)
        empty_cells = np.count_nonzero(game.board == 0)
        elapsed = end_time - start_time

        print(f"第 {step+1} 步 | 动作 {action} | 得分 {game.score} | 最大数字 {max_tile} | 空格 {empty_cells} | 耗时 {elapsed:.2f} 秒")
        print_board(game.board)
