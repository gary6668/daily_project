# 这里可以放调试工具，比如打印棋盘、美化输出
def print_board(board):
    print("\n".join([" ".join([f"{v:4d}" for v in row]) for row in board]))
    print("-" * 20)
