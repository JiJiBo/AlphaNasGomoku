# 棋盘
class GomokuBoard:
    def __init__(self, size=19, count_win=5, ):
        """
        初始化函数
        :param size: 棋盘大小
        :param count_win: 多少子能赢
        """
        self.size = size
        self.count_win = count_win
