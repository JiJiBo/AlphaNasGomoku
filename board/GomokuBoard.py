# 棋盘
import copy
from typing import Optional

import numpy as np

from board.GomukuPlayer import PLAYER_EMPTY, Winner, PLAYER_WHITE, PLAYER_BLACK


class GomokuAction:
    def __init__(self, x, y, flag):
        self.x = x
        self.y = y
        self.flag = flag

    def is_available(self, board: Optional['GomokuBoard']):
        # 检测位置
        if self.x < 0 or self.x > board.size or self.y < 0 or self.y > board.size:
            # print("检测位置")
            return False
        # 检测是否为空白处
        if board.board[self.x][self.y] != PLAYER_EMPTY:
            # print(f"不为空白处 {self.x}, {self.y}  {board.board[self.x][self.y]}")
            # for ac in  board.available():
            #     print(f"{ac} {board.board[ac[0]][ac[1]]} ")
            return False
        # 检测是否是该他下子
        if self.flag == board.last_move().flag:
            # print("不该他下子")
            return False
        return True


class GomokuBoard:
    def __init__(self, size=19, count_win=5, ):
        """
        初始化函数
        :param size: 棋盘大小
        :param count_win: 多少子能赢
        """
        self.size = size
        self.count_win = count_win
        # 棋盘
        self.board = np.zeros((size, size))
        # 走的步数
        self.move_count = 0
        # 走的历史
        self.history = []  # [(x,y,player_flag)]
        self.win_path = None

    def last_move(self) -> GomokuAction:
        if len(self.history) == 0:
            return GomokuAction(-1, -1, PLAYER_EMPTY)
        else:
            return self.history[-1]

    def reset(self):
        """
        重置棋盘
        :return:
        """
        # 棋盘
        self.board = np.zeros((self.size, self.size))
        # 走的步数
        self.move_count = 0
        # 走的历史
        self.history = []

    def available(self):
        """
        获取空白位置
        :return: 空白位置
        """
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                move = self.board[x][y]
                if move == PLAYER_EMPTY:
                    moves.append((x, y))
        return moves

    def get_winner(self) -> Winner:
        """
            得到获胜方
        :return: 获胜方
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for x in range(self.size):
            for y in range(self.size):
                flag = self.board[x][y]
                if flag == PLAYER_EMPTY:
                    continue

                for dir_x, dir_y in directions:
                    count = 0
                    seq = []
                    for i in range(self.count_win):
                        nx = x + dir_x * i
                        ny = y + dir_y * i
                        seq.append([nx, ny])

                        if 0 <= nx < self.size and 0 <= ny < self.size:
                            if self.board[nx][ny] == flag:
                                count += 1
                                if count >= self.count_win:
                                    self.win_path = seq[:self.count_win]
                                    if flag == PLAYER_WHITE:
                                        return Winner.WHITE
                                    elif flag == PLAYER_BLACK:
                                        return Winner.BLACK

        return Winner.EMPTY

    def is_terminal(self) -> bool:
        """
        游戏是否结束
        :return: 是否结束
        """
        return len(self.available()) == 0 or self.get_winner() != Winner.EMPTY

    def get_score(self) -> int:
        """
        结束后的打分
        :return: 打分
        """
        if self.is_terminal():
            flag = self.get_winner().value.real
            return flag
        else:
            return 0

    def step(self, action: GomokuAction) -> [np.ndarray, Winner]:
        """
        合法下棋
        :param action:落子信息
        :return:棋盘，状态
        """
        if not action.is_available(self):
            raise RuntimeError("违法落子")
        self.move_count += 1
        self.history.append(action)
        self.board[action.x, action.y] = action.flag
        return self.board, self.get_winner()

    def copy(self) -> Optional['GomokuBoard']:
        """
        拷贝棋盘
        :return: 棋盘
        """
        new_board = GomokuBoard(self.size, self.count_win)
        new_board.board = copy.deepcopy(self.board)
        new_board.history = copy.deepcopy(self.history)
        new_board.move_count = self.move_count
        return new_board

    def get_planes_3ch(self, flag):
        b = self.board
        me = (b == flag).astype(np.float32)
        opp = (b == -flag).astype(np.float32)
        empty = (b == 0).astype(np.float32)
        return np.stack([me, opp, empty], axis=0).astype(np.float32)
