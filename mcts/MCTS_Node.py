from dataclasses import dataclass
from typing import Optional, Dict, List

from board.GomokuBoard import GomokuBoard, GomokuAction


@dataclass
class Edge:
    child: Optional['MCTS_Node', None]
    prior: float


class MCTS_Node:
    def __init__(self, board: GomokuBoard, player: int, parent: [Optional['MCTS_Node'], None] = None,
                 prior_p: float = 1.0):
        self.board = board
        self.player = player
        self.parent = parent
        # 值
        # 子节点
        self.children: Dict[GomokuAction,Edge] = {}  # 动作 -> 子节点(node,prior)

        # MCTS 统计
        self.wins_value = 0.0  # 累积价值 (Q 部分的分子)
        self.visits = 0  # 访问次数 (Q 部分的分母)

        # PUCT 相关
        self.prior_p = prior_p  # 来自策略网络的先验概率 P(s,a)

    def update(self, value: float):
        """回溯时更新本节点的价值统计"""
        self.visits += 1
        self.wins_value += value
