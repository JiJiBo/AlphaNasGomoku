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
    def __init__(self, model: PolicyValueNet, device=None, use_rand=0.03, c_puct=1.4, tau=0.9):
        self.model = model
        self.use_rand = use_rand
        self.c_puct = c_puct
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.visit_nodes: List['MCTS_Node'] = []
        self.tau = tau

    def run(self, root_board: GomokuBoard, player: int, number_samples=100, is_train=False, cur_root=None):
        if cur_root is None:
            root_node = MCTS_Node(root_board, player)
        else:
            root_node = cur_root
        self.visit_nodes.append(root_node)
        for _ in range(number_samples):
            node = root_node
            search_path = [node]
            while node.children:
                node = self.select_child(node)
                search_path.append(node)
            if not node.board.is_terminal():
                self.expand(node)
            value = node.board.get_score() * node.player
            for n in reversed(search_path):
                n.update(value)
                value = -value
        return self.get_result_action(root_node, is_train=is_train), root_node

    def select_child(self, node: MCTS_Node):
        # 得到当前节点的访问次数
        total_visits = sum([(edg.child.visits if edg.child is not None else 0) for edg in node.children.values()])
        # 开平方
        explore_buff = math.sqrt(total_visits)

        best_score = -float('inf')
        best_child = None
        best_move = None
        best_edg = None
        # 获取 最优信息
        for action, edg in node.children.items():
            if not action.is_available(node.board):
                continue
            child, prior = edg.child, edg.prior
            vis_count = child.visits if child else 0
            Q = child.total_value / vis_count if vis_count > 0 else 0
            U = self.c_puct * prior * explore_buff / (vis_count + 1)
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = child
                best_move = action
                best_edg = edg
        # 如果best child 是 空的，就给他赋值
        if best_child is None and best_move is not None:
            # 把 父节点 拷贝 一份
            new_board = node.board.copy()
            # 走下 关键的 最优一步
            new_board.step(best_move)
            # 给 节点 赋值
            # flag 要 取反
            # 最关键 的 是 prior=best_edg.prior 因为 这 涉及到 一个 约定，可见 注释
            best_child = MCTS_Node(new_board, -best_move.flag, parent=node, prior=best_edg.prior)
            # 给其 赋值
            node.children[best_move] = Edge(best_child, best_edg.prior)
        return best_child

    def expand(self, node: MCTS_Node):
        # 得到先验概率（ 已经  log_softmax 处理 ）和胜率
        policy_logits, value = self.model.calc_one_board(torch.from_numpy(node.board.get_planes_3ch(node.player)))
        # 得到 空白位置
        moves = node.board.available()
        # 筛选可用的（空白位置的）先验概率
        priors = np.array([policy_logits[x, y] for x, y in moves], dtype=np.float32)
        # 求和，为先验概率归一化做准备
        s = priors.sum()
        # 对 先验概率 -> priors  进行归一化 。如果s不大于0， 先验概率 为 1/len(可用位置)   但是肯定进行了归一化
        priors = priors / s if s > 0 else np.ones(len(moves), dtype=np.float32) / max(1, len(moves))
        # 对 先验概率 进行 扰动
        epsilon = 0.25
        alpha = 0.03
        dir_noise = np.random.dirichlet([alpha] * len(moves))
        priors = [(1 - epsilon) * p + epsilon * n for p, n in zip(priors, dir_noise)]
        # 再次 求出 先验概率 的 和
        ps = sum(priors)
        # 再次 归一化 得到 最终 的  priors -> 先验概率
        priors = [p / ps if ps > 0 else 1.0 / len(priors) for p in priors]

        for (x, y), p in zip(moves, priors):
            node.children[GomokuAction(x, y, node.player)] = Edge(None, float(p))
        # 最终 总结
        # 经过 expand 神经网络 的 输出，胜率 赋予 了 根节点 。 动作 赋予 了 子节点 的 key 。
        # edge 的 prior 是 下子 的 概率 ， 不是 胜率。
        node.wins_value = float(value)

    def get_result_action(self, node: MCTS_Node, is_train=False) -> tuple[GomokuAction, np.ndarray]:
        return node.best_action(self.tau if is_train else 0, node.board.available())

    def get_train_data(self, winner: int):
        boards, policies, values, weights = [], [], [], []
        train_buff = 0.8
        train_simulation = 100
        for node in self.visit_nodes:
            # 节点 玩家 视角下 的 最终 对局 结果
            z = winner if node.player == GomokuBoard.PLAYER_BLACK else -winner
            # 获取 所有 孩子的 访问总数
            total_visits = sum([(edg.child.visits if edg.child is not None else 0) for edg in node.children.values()])
            # 得到 这个 节点的 动作 策略 :: 已经 归一化 了
            pi = node.get_train()
            # 得到 棋盘 数据
            boards.append(node.board.copy().get_planes_3ch(node.player))
            policies.append(pi)
            # 得到 当前节点 的 价值
            values.append(float(z))
            # 得到 每条 训练数据 的 重要性 系数
            weights.append(total_visits / train_simulation * train_buff)
        return self.augment_data(boards, policies, values, weights)

    def augment_data(self, boards, policies, values, weights):
        boards = torch.from_numpy(np.array(boards)).float()
        policies = torch.from_numpy(np.array(policies)).float()
        values = torch.from_numpy(np.array(values)).float()
        weights = torch.from_numpy(np.array(weights)).float()

        augmented_boards = []
        augmented_policies = []
        augmented_values = []
        augmented_weights = []

        for board, policy, value, weight in zip(boards, policies, values, weights):
            value = value.detach().clone().float()
            weight = weight.detach().clone().float()
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

    def show_nn(self, board: GomokuBoard):
        """
        展示热度图
        返回策略概率和价值评估的热度图
        """
        board_size = board.size

        # 获取当前棋盘的策略概率
        prob, value = self.model.calc_one_board(torch.from_numpy(board.get_planes_3ch(1)))
        print("===")
        print(prob.max(), prob.min())
        print(value)
        # 初始化价值矩阵
        val = np.zeros((board_size, board_size), dtype=np.float32)

        # 对每个位置进行评估
        for i in range(board_size):
            for j in range(board_size):
                if board.board[i][j] != 0:  # 如果位置已有棋子，概率设为0
                    prob[i][j] = 0
                else:
                    # 创建新棋盘，在当前位置下子
                    new_board = board.copy()
                    new_board.step(
                        GomokuAction(i, j,
                                     board.last_move().flag if board.last_move().flag != 0 else GomokuBoard.PLAYER_BLACK))

                    # 翻转棋盘颜色进行评估
                    flipped_board = new_board.copy()
                    for x in range(board_size):
                        for y in range(board_size):
                            if flipped_board.board[x][y] != 0:
                                flipped_board.board[x][y] *= -1

                    # 获取翻转后棋盘的价值评估
                    planes = flipped_board.get_planes_3ch(-board.last_move().flag)
                    _, val[i][j] = self.model.calc_one_board(
                        planes)
                    val[i][j] = float(-val[i][j])  # 取负值，因为是从对手视角评估

        return prob, val
