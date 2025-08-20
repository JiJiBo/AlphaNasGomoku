import numpy as np
from board.GomokuBoard import GomokuBoard, GomokuAction, PLAYER_BLACK, PLAYER_WHITE, PLAYER_EMPTY, Winner
from train import generate_random_safe_board

# -------------------------------
# 测试
# -------------------------------
if __name__ == "__main__":
    b = generate_random_safe_board(board_size=15)
    print("随机非终局棋盘生成完成")
    print("棋盘步数:", b.move_count)
    print("是否终局:", b.is_terminal())
    print("空位数:", len(b.available()))
    print(b.board)
