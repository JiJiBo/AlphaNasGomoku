import copy
import math
import random
from typing import List

import numpy as np
import torch
from numpy import ndarray
import tqdm

from board import GomokuBoard
from board.GomokuBoard import GomokuAction
from mcts.MCTS_Node import MCTS_Node, Edge
from net.GomokuNet import PolicyValueNet


class MCTS_Agent:
    def __init__(self, model: PolicyValueNet, device=None, use_rand=0.1, c_puct=1.4):
        self.model = model
        self.use_rand = use_rand
        self.c_puct = c_puct
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.visit_nodes: List['MCTS_Node'] = []

    def run(self, root_board: GomokuBoard, player: int, number_samples=800, is_train=False):
        root_node = MCTS_Node(root_board, player)
        self.visit_nodes.append(root_node)
        for _ in tqdm.trange(number_samples):
            node = root_node
            search_path = [node]
            while node.children:
                node = self.select_child(node)
                search_path.append(node)
            if node.board.is_terminal():
                value = node.board.get_score()
            else:
                value = self.expand(node)
            for n in reversed(search_path):
                n.update(value)
                value = -value
        return self.get_result_action(root_node, is_train=is_train)

    def select_child(self, node: MCTS_Node):
        total_visits = sum([(edg.child.visits if edg.child is not None else 0) for edg in node.children.values()])
        explore_buff = math.pow(total_visits, 0.5)
        best_move: [GomokuAction, None] = None
        best_score = -float('inf')
        best_child: [MCTS_Node, None] = None
        best_edg = None
        for action, edg in node.children.items():
            if not action.is_available(node.board):
                continue
            child, prior = edg.child, edg.prior
            vis_count = 0
            Q = 0
            if child is not None:
                vis_count = child.visits
                if vis_count != 0:
                    Q = total_visits / vis_count
            U = self.c_puct * prior * explore_buff / (vis_count + 1)
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = child
                best_move = action
                best_edg = edg
        if best_child is None and best_move is not None:
            # 使用深拷贝，保证每个节点棋盘独立
            new_board = node.board.copy()
            new_board.step(best_move)
            best_child = MCTS_Node(new_board, -best_move.flag, parent=node, prior_p=best_edg.prior)
            node.children[best_move] = Edge(best_child, best_edg.prior)
        return best_child

    def expand(self, node: MCTS_Node) -> float:
        policy_logits, value = self.model.calc_one_board(
            torch.from_numpy(node.board.get_planes_3ch())
        )
        moves = node.board.available()
        priors = np.array([policy_logits[x, y] for x, y in moves], dtype=np.float32)
        s = float(priors.sum())
        if s > 0:
            priors /= s
        else:
            priors = np.ones(len(moves), dtype=np.float32) / max(1, len(moves))
        priors = [max(0.0, p + random.normalvariate(0, self.use_rand)) for p in priors]
        ps = float(sum(priors))
        priors = [p / ps if ps > 0 else 1.0 / len(priors) for p in priors]
        for (y, x), p in zip(moves, priors):
            node.children[GomokuAction(x, y, node.player)] = Edge(None, float(p))

        return float(value)

    def get_result_action(self, node: MCTS_Node, is_train=False) -> tuple[GomokuAction, ndarray]:
        return node.best_action(1.2 if is_train else 0, node.board.available())

    def get_train_data(self):
        boards, policies, values, weights = [], [], [], []
        train_buff = 0.8
        train_simulation = 30
        for node in self.visit_nodes:
            total_visits = sum([(edg.child.visits if edg.child is not None else 0) for edg in node.children.values()])
            best_move, pi = node.best_action(1.2, node.board.available())
            boards.append(node.board.copy().get_planes_3ch())
            policies.append(pi)
            values.append(node.board.get_score())
            weights.append(math.sqrt(total_visits / train_simulation) * train_buff)
        return self.augment_data(boards, policies, values, weights)

    def augment_data(self, boards, policies, values, weights):
        augmented_boards = []
        augmented_policies = []
        augmented_values = []
        augmented_weights = []

        for board, policy, value, weight in zip(boards, policies, values, weights):
            value = torch.tensor(value).clone().detach().float()
            weight = torch.tensor(weight).clone().detach().float()

            # D4 对称变换
            for k in range(4):
                for o in range(2):
                    new_board = torch.rot90(board, k, [1, 2])
                    new_policy = torch.rot90(policy, k, [0, 1])
                    if o:
                        new_board = torch.flip(new_board, [2])
                        new_policy = torch.flip(new_policy, [1])

                    augmented_boards.append(new_board)
                    augmented_policies.append(new_policy)
                    augmented_values.append(value)
                    augmented_weights.append(weight)

        return (
            torch.stack(augmented_boards),
            torch.stack(augmented_policies),
            torch.stack(augmented_values),
            torch.stack(augmented_weights)
        )
