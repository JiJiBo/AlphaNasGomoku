from typing import List

from board import GomokuBoard
from mcts.MCTS_Node import MCTS_Node
from net.GomokuNet import PolicyValueNet


class MCTSAgent:
    def __init__(self, model: PolicyValueNet, device, use_rand=0.1, c_puct=1.4):
        self.model = model
        self.use_rand = use_rand
        self.c_puct = c_puct
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.visit_nodes: List['MCTS_Node'] = []

    def run(self, root_board: GomokuBoard, player: int, number_samples=100, is_train=False):
        root_node = MCTS_Node(root_board, player)
