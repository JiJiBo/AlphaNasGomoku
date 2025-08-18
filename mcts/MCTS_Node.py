from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np

from board.GomokuBoard import GomokuBoard, GomokuAction


@dataclass
class Edge:
    child: Optional['MCTS_Node']
    # 特殊处理 prior 等于 child 的 prior
    prior: float


class MCTS_Node:
    def __init__(self, board: GomokuBoard, player: int, parent: Optional['MCTS_Node'] = None,
                 prior: float = 1.0):
        self.board = board
        self.player = player
        self.parent = parent
        self.children: Dict[GomokuAction, Edge] = {}
        self.wins_value = 0.0
        self.visits = 0
        self.prior = prior

    def update(self, value: float):
        self.visits += 1
        self.wins_value = value

    def best_action(self, tau: float = 0.0, legal_actions: Optional[List[int]] = None,
                    alpha: float = 0.1, rng: Optional[np.random.Generator] = None) -> Tuple[GomokuAction, np.ndarray]:
        # 获取 一个 选择器
        if rng is None:
            rng = np.random.default_rng()
        # 得到 所有 孩子 的 动作
        actions = list(self.children.keys())
        # 筛选 可用的 动作集合
        if legal_actions is not None:
            actions = [a for a in actions if (a.x, a.y) in legal_actions]
        if not actions:
            raise ValueError("没有合法动作可选择")
        # 得到 所有 孩子的 访问次数 集合
        visits = np.array([self.children[a].child.visits if self.children[a].child else 0 for a in actions],
                          dtype=np.float64)
        # 与其 对应的 访问概率
        priors = np.array([self.children[a].prior for a in actions], dtype=np.float64)
        # 看看是不是训练模式
        if tau <= 1e-6:
            # 不是训练
            # 得到 最大 访问次数
            max_visits = visits.max()
            # 取出 访问 最多的 点
            candidates = [a for a, v in zip(actions, visits) if v == max_visits]
            if len(candidates) > 1:
                # 如果 不止 一个 访问 次数 第一名
                # tie-breaker using prior
                # 得到 先验概率 最大的 点
                best_prior = max(self.children[a].prior for a in candidates)
                candidates = [a for a in candidates if self.children[a].prior == best_prior]
            # 得到 最优 移动点
            ### TODO 这里存疑，看看是不是应该根据选择的次数来判定最优点
            best_move = rng.choice(candidates)
            pi = np.zeros(len(actions), dtype=np.float64)
            pi[actions.index(best_move)] = 1.0
            return best_move, pi

        # tau > 0, 将访问次数与 prior 混合
        # 训练模式
        #
        w = (visits + alpha * priors) ** (1.0 / tau)
        pi = w / w.sum()
        best_move = rng.choice(actions, p=pi)
        return best_move, pi

    def get_train(self, alpha: float = 0.1) -> np.ndarray:
        # 初始化 一个 策略 棋盘
        policy = np.zeros((self.board.size, self.board.size), dtype=np.float64)
        total_visits = sum(edge.child.visits if edge.child else 0 for edge in self.children.values())
        total_prior = sum(edge.prior for edge in self.children.values())
        # 遍历所有的孩子
        for move, edge in self.children.items():
            # 得到 孩子的 访问次数
            visits = edge.child.visits if edge.child else 0
            # 给 这个 位置 下棋的 概率 赋值
            policy[move.x, move.y] = visits + alpha * edge.prior
        # 归一化 策略
        policy /= policy.sum() if policy.sum() > 0 else 1
        # 返回 这个 节点的 动作 策略 :: 已经 归一化 了
        # 形状 是 (self.board.size, self.board.size) -> like  (19,19)
        return policy
