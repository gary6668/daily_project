# main.py
from game import Game2048

g = Game2048()

while True:
    print("棋盘：\n", g.board)
    print("得分：", g.score)
    move = input("Move (w=上, a=左, s=下, d=右, q=退出): ")
    mapping = {'w': 0, 'd': 1, 's': 2, 'a': 3}
    if move in mapping:
        g.move(mapping[move])
    elif move == 'q':
        break
    else:
        print("无效输入，请重新输入")
