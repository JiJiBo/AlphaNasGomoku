from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np

from board.GomokuBoard import GomokuBoard, GomokuAction


@dataclass
class Edge:
    child: Optional['MCTS_Node']
    prior: float


class MCTS_Node:
    def __init__(self, board: GomokuBoard, player: int, parent: Optional['MCTS_Node'] = None,
                 prior_p: float = 1.0):
        self.board = board
        self.player = player
        self.parent = parent
        # 值
        # 子节点
        self.children: Dict[GomokuAction, Edge] = {}  # 动作 -> 子节点(node,prior)

        # MCTS 统计
        self.wins_value = 0.0  # 累积价值 (Q 部分的分子)
        self.visits = 0  # 访问次数 (Q 部分的分母)

        # PUCT 相关
        self.prior_p = prior_p  # 来自策略网络的先验概率 P(s,a)

    def update(self, value: float):
        """回溯时更新本节点的价值统计"""
        self.visits += 1
        self.wins_value += value

    def best_action(self, tau: float = 0.0, legal_actions: Optional[List[int]] = None,
                    rng: Optional[np.random.Generator] = None) -> Tuple[GomokuAction, np.ndarray]:
        """
        获取当前节点的最佳动作。
        参数:
            tau: 温度, 训练期 >0, 实战期 0
            legal_actions: 可选，只考虑合法动作
            rng: np.random.Generator, 可用于采样
        返回:
            best_move: 选中的动作
            pi: 所有动作的策略分布 (顺序与 self.children.keys() 对应)
        """
        if rng is None:
            rng = np.random.default_rng()

        # 筛选合法动作
        actions = list(self.children.keys())
        if legal_actions is not None:
            actions = [a for a in actions if (a.x, a.y) in legal_actions]
        # print(len(actions))
        # print("==0==")
        # for ac in actions:
        #     print(f"{ac.x} {ac.y} {self.board.board[ac.x][ac.y]} ")
        #
        # print("==1==")
        if not actions:
            raise ValueError("没有合法动作可选择")

        # 统计访问次数
        visits = np.array([self.children[a].child.visits if self.children[a].child else 0 for a in actions],
                          dtype=np.float64)

        # 温度为0时，贪心选择
        if tau <= 1e-6:
            max_visits = visits.max()
            candidates = [a for a, v in zip(actions, visits) if v == max_visits]
            best_move = rng.choice(candidates)
            # print("== == ",best_move.x, best_move.y)
            # 构造 one-hot 策略分布
            pi = np.zeros(len(actions), dtype=np.float64)
            pi[actions.index(best_move)] = 1.0
            return best_move, pi

        # 训练期，温度抽样
        eps = 1e-12
        w = np.power(np.maximum(visits, eps), 1.0 / tau)
        pi = w / w.sum()
        best_move = rng.choice(actions, p=pi)
        return best_move, pi

    def get_train(self, ) -> Tuple[GomokuAction, np.ndarray]:
        """
        获取当前节点的最佳动作。
        参数:
            tau: 温度, 训练期 >0, 实战期 0
            legal_actions: 可选，只考虑合法动作
        返回:
            pi:
        """
        policy = np.zeros((self.board.size, self.board.size), dtype=np.float64)
        for move, edge in self.children.items():
            policy[move.x, move.y] = edge.prior
        return policy
