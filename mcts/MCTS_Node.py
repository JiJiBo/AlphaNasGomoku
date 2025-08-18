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
        self.wins_value += value

    def best_action(self, tau: float = 0.0, legal_actions: Optional[List[int]] = None,
                    alpha: float = 0.1, rng: Optional[np.random.Generator] = None) -> Tuple[GomokuAction, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        actions = list(self.children.keys())
        if legal_actions is not None:
            actions = [a for a in actions if (a.x, a.y) in legal_actions]
        if not actions:
            raise ValueError("没有合法动作可选择")

        visits = np.array([self.children[a].child.visits if self.children[a].child else 0 for a in actions], dtype=np.float64)
        priors = np.array([self.children[a].prior for a in actions], dtype=np.float64)

        if tau <= 1e-6:
            max_visits = visits.max()
            candidates = [a for a, v in zip(actions, visits) if v == max_visits]
            if len(candidates) > 1:
                # tie-breaker using prior
                best_prior = max(self.children[a].prior for a in candidates)
                candidates = [a for a in candidates if self.children[a].prior == best_prior]
            best_move = rng.choice(candidates)
            pi = np.zeros(len(actions), dtype=np.float64)
            pi[actions.index(best_move)] = 1.0
            return best_move, pi

        # tau > 0, 将访问次数与 prior 混合
        w = (visits + alpha * priors) ** (1.0 / tau)
        pi = w / w.sum()
        best_move = rng.choice(actions, p=pi)
        return best_move, pi

    def get_train(self, alpha: float = 0.1) -> np.ndarray:
        policy = np.zeros((self.board.size, self.board.size), dtype=np.float64)
        total_visits = sum(edge.child.visits if edge.child else 0 for edge in self.children.values())
        total_prior = sum(edge.prior for edge in self.children.values())
        for move, edge in self.children.items():
            visits = edge.child.visits if edge.child else 0
            policy[move.x, move.y] = visits + alpha * edge.prior
        policy /= policy.sum() if policy.sum() > 0 else 1
        return policy
