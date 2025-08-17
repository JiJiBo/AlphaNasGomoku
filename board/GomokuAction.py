from board.GomokuBoard import GomokuBoard
from board.GomukuPlayer import PLAYER_EMPTY


class GomokuAction:
    def __init__(self, x, y, flag):
        self.x = x
        self.y = y
        self.flag = flag

    def is_available(self, board: GomokuBoard):
        # 检测位置
        if self.x < 0 or self.x > board.size.x or self.y < 0 or self.y > board.size.y:
            return False
        # 检测是否为空白处
        if board.board[self.x][self.y] != PLAYER_EMPTY:
            return False
        # 检测是否是该他下子
        if self.flag == board.last_move().flag:
            return False
        return True
