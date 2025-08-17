from typing import Optional, Dict

from board.GomokuBoard import GomokuBoard


class MCTS_Node:
    def __init__(self, board: GomokuBoard, player: int, parent: [Optional['MCTS_Node'], None] = None,
                 prior_p: float = 1.0):
        self.board = board
        self.player = player
        self.parent = parent
        # 值
        # 子节点
        self.children: Dict[int, 'MCTS_Node'] = {}  # 动作 -> 子节点

        # MCTS 统计
        self.wins_value = 0.0  # 累积价值 (Q 部分的分子)
        self.visits = 0  # 访问次数 (Q 部分的分母)

        # PUCT 相关
        self.prior_p = prior_p  # 来自策略网络的先验概率 P(s,a)

    @property
    def Q(self) -> float:
        """平均价值 Q(s,a)"""
        if self.visits == 0:
            return 0.0
        return self.wins_value / self.visits

    def U(self, c_puct: float) -> float:
        """探索价值 U(s,a)"""
        if self.parent is None:
            return 0.0
        return c_puct * self.prior_p * ((self.parent.visits ** 0.5) / (1 + self.visits))

    def get_value(self, c_puct: float) -> float:
        """PUCT 选择公式 = Q + U"""
        return self.Q + self.U(c_puct)

    def update(self, value: float):
        """回溯时更新本节点的价值统计"""
        self.visits += 1
        self.wins_value += value
