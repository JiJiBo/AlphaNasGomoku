import os
import sys
import time
import random
import numpy as np
import pygame
import torch
from typing import Optional

from board.GomokuBoard import GomokuBoard, GomokuAction
from mcts.MCTS_Agent import MCTS_Agent
from net.GomokuNet import PolicyValueNet


class Agent:
    """Base agent interface."""

    def select_move(self, board: GomokuBoard, player: int):
        raise NotImplementedError


class RandomAgent(Agent):
    """Agent that plays random legal moves."""

    def select_move(self, board: GomokuBoard, player: int):
        moves = board.available()
        return random.choice(moves)


class MCTSAgent(Agent):
    """Agent that uses MCTS with a neural network model."""

    def __init__(self, model: PolicyValueNet, simulations: int = 100):
        self.mcts = MCTS_Agent(model)
        self.simulations = simulations

    def select_move(self, board: GomokuBoard, player: int):
        info, root = self.mcts.run(board, player, self.simulations, is_train=False)
        action, probs = info
        # self.mcts.get_train_data()
        print(f"probs: {probs.max()}")
        return (action.x, action.y)


class ModelAgent(MCTSAgent):
    """Agent that loads a PolicyValueNet model from a checkpoint."""

    def __init__(
            self,
            board_size: int = 19,
            device: Optional[str] = None,
            simulations: int = 100,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PolicyValueNet(board_size=board_size)
        # state = torch.load(model_path, map_location=device)
        # model.load_state_dict(state)
        model.to(device)
        model.eval()
        super().__init__(model, simulations)


class PygameMatch:
    """Display a real-time match between two agents or a human using pygame."""

    def __init__(self, agent_black: Optional[Agent], agent_white: Optional[Agent],
                 board_size: int = 19, cell_size: int = 40, margin: int = 20,
                 delay: int = 100):
        self.board = GomokuBoard(board_size)
        self.agent_black = agent_black
        self.agent_white = agent_white
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = margin
        self.delay = delay

        # 添加神经网络分析相关属性
        self.show_nn_analysis = True  # 默认开启神经网络分析
        self.nn_prob = None
        self.nn_val = None
        self.current_agent = None  # 当前AI代理

        pygame.init()
        size = margin * 2 + cell_size * (board_size - 1)
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Gomoku Match")

        # 如果有AI代理，自动进行第一次神经网络分析
        if self.agent_black and hasattr(self.agent_black, 'mcts'):
            self.current_agent = self.agent_black
            try:
                self.nn_prob, self.nn_val = self.current_agent.mcts.show_nn(self.board)
                print("神经网络分析已默认开启")
                print(f"策略概率范围: {self.nn_prob.min():.4f} ~ {self.nn_prob.max():.4f}")
                print(f"价值评估范围: {self.nn_val.min():.4f} ~ {self.nn_val.max():.4f}")
            except Exception as e:
                print(f"初始神经网络分析失败: {e}")
                self.show_nn_analysis = False
        elif self.agent_white and hasattr(self.agent_white, 'mcts'):
            self.current_agent = self.agent_white
            try:
                self.nn_prob, self.nn_val = self.current_agent.mcts.show_nn(self.board)
                print("神经网络分析已默认开启")
                print(f"策略概率范围: {self.nn_prob.min() :.4f} ~ {self.nn_prob.max() :.4f}")
                print(f"价值评估范围: {self.nn_val.min():.4f} ~ {self.nn_val.max():.4f}")
            except Exception as e:
                print(f"初始神经网络分析失败: {e}")
                self.show_nn_analysis = False

    # ---- helpers ----
    def is_human_turn(self, player: int) -> bool:
        return (player == 1 and self.agent_black is None) or (
                player == -1 and self.agent_white is None
        )

    def get_human_move(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_u:
                    return 'undo'
                if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    # 按N键触发神经网络分析
                    self.toggle_nn_analysis()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                    # 按H键显示帮助信息
                    self.show_help()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    x = int(round((mx - self.margin) / self.cell_size))
                    y = int(round((my - self.margin) / self.cell_size))
                    if 0 <= x < self.board_size and 0 <= y < self.board_size:
                        if self.board.board[y, x] == 0:
                            return (y, x)
            pygame.time.wait(50)

    def toggle_nn_analysis(self):
        """切换神经网络分析显示"""
        if self.current_agent and hasattr(self.current_agent, 'mcts'):
            try:
                self.nn_prob, self.nn_val = self.current_agent.mcts.show_nn(self.board)
                self.show_nn_analysis = not self.show_nn_analysis
                print(f"神经网络分析: {'开启' if self.show_nn_analysis else '关闭'}")
                if self.show_nn_analysis:
                    print(f"策略概率范围: {self.nn_prob.min():.4f} ~ {self.nn_prob.max():.4f}")
                    print(f"价值评估范围: {self.nn_val.min():.4f} ~ {self.nn_val.max():.4f}")
            except Exception as e:
                print(f"神经网络分析失败: {e}")
                self.show_nn_analysis = False
        else:
            print("当前没有可用的AI代理进行神经网络分析")

    def get_nn_color(self, value, min_val, max_val, is_policy=True):
        """根据神经网络输出值获取颜色"""
        if is_policy:
            # 策略概率：用绿色热度图，概率越大颜色越深
            if value == 0:  # 已有棋子的位置
                return (128, 128, 128)  # 灰色
            normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            normalized = max(0, min(1, normalized))
            # 绿色热度图：从浅绿到深绿
            green_intensity = int(50 + normalized * 205)  # 50-255的绿色值
            return (0, green_intensity, 0)
        else:
            # 价值评估：从蓝色（负值）到红色（正值）
            if value == 0:  # 已有棋子的位置
                return (128, 128, 128)  # 灰色
            normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            normalized = max(0, min(1, normalized))
            if normalized < 0.5:
                # 蓝色到白色
                intensity = int(255 * (0.5 - normalized) * 2)
                return (intensity, intensity, 255)
            else:
                # 白色到红色
                intensity = int(255 * (normalized - 0.5) * 2)
                return (255, intensity, intensity)

    def show_winner(self, winner: int):
        font = pygame.font.SysFont(None, 48)
        if winner == 1:
            text = "Black wins"
        elif winner == -1:
            text = "White wins"
        else:
            text = "Draw"
        img = font.render(text, True, (255, 0, 0))
        rect = img.get_rect(center=(self.screen.get_width() // 2, self.margin // 2))
        self.screen.blit(img, rect)
        pygame.display.flip()

    def draw_board(self):
        self.screen.fill((205, 170, 125))
        start = self.margin
        end = self.margin + self.cell_size * (self.board_size - 1)

        # 绘制棋盘网格
        for i in range(self.board_size):
            offset = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, (0, 0, 0), (start, offset), (end, offset), 1)
            pygame.draw.line(self.screen, (0, 0, 0), (offset, start), (offset, end), 1)

        # 如果开启神经网络分析，绘制热度图
        if self.show_nn_analysis and self.nn_prob is not None and self.nn_val is not None:
            # 获取概率前10的位置和概率大于55%的位置
            prob_data = []
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if self.board.board[y, x] == 0:  # 只考虑空白位置
                        prob_data.append((y, x, self.nn_prob[y, x]))
            
            # 按概率排序，获取前10
            prob_data.sort(key=lambda x: x[2], reverse=True)
            top_10_positions = set((y, x) for y, x, _ in prob_data[:10])
            
            # 获取概率大于55%的位置
            high_prob_threshold = 0.55
            high_prob_positions = set()
            for y, x, prob in prob_data:
                if prob > high_prob_threshold:
                    high_prob_positions.add((y, x))
            
            # 合并需要显示的位置
            show_positions = top_10_positions.union(high_prob_positions)
            
            # 计算概率范围用于颜色映射
            if prob_data:
                prob_min, prob_max = min(p[2] for p in prob_data), max(p[2] for p in prob_data)
                
                # 绘制绿色热度图
                for y, x, prob in prob_data:
                    if (y, x) in show_positions:
                        # 用绿色热度图，概率越大颜色越深
                        green_intensity = int(50 + (prob - prob_min) / (prob_max - prob_min) * 205) if prob_max != prob_min else 127
                        color = (0, green_intensity, 0)
                        
                        # 绘制半透明的绿色热度方块
                        rect = pygame.Rect(
                            self.margin + x * self.cell_size - self.cell_size // 2,
                            self.margin + y * self.cell_size - self.cell_size // 2,
                            self.cell_size, self.cell_size
                        )
                        s = pygame.Surface((self.cell_size, self.cell_size))
                        s.set_alpha(120)  # 半透明
                        s.fill(color)
                        self.screen.blit(s, rect)

        # 绘制棋子
        for y in range(self.board_size):
            for x in range(self.board_size):
                p = self.board.board[y, x]
                if p != 0:
                    color = (0, 0, 0) if p == 1 else (255, 255, 255)
                    pos = (self.margin + x * self.cell_size, self.margin + y * self.cell_size)
                    pygame.draw.circle(self.screen, color, pos, self.cell_size // 2 - 2)

        # 绘制获胜路径
        if self.board.win_path:
            pts = [(
                self.margin + x * self.cell_size,
                self.margin + y * self.cell_size
            ) for y, x in self.board.win_path]
            pygame.draw.lines(self.screen, (255, 0, 0), False, pts, 3)

        # 绘制神经网络分析状态指示
        if self.show_nn_analysis:
            font = pygame.font.SysFont(None, 24)
            text = font.render("NN Analysis: ON", True, (255, 0, 0))
            self.screen.blit(text, (10, 10))
            
            # 显示胜率信息
            if self.nn_val is not None:
                # 获取当前局面的价值评估（胜率）
                current_value = self.nn_val.mean()  # 使用平均值作为整体胜率
                # 将价值转换为胜率百分比 (假设价值范围是-1到1，转换为0-100%)
                win_rate = (current_value + 1) * 50  # 转换为0-100%
                
                # 显示胜率
                win_rate_text = f"胜率: {win_rate:.1f}%"
                if current_value > 0:
                    # 黑方优势
                    win_rate_text += " (黑方优势)"
                    color = (0, 0, 0)
                elif current_value < 0:
                    # 白方优势
                    win_rate_text += " (白方优势)"
                    color = (255, 255, 255)
                else:
                    # 均势
                    win_rate_text += " (均势)"
                    color = (128, 128, 128)
                
                # 在棋盘上方显示胜率
                win_rate_surface = font.render(win_rate_text, True, color)
                win_rate_rect = win_rate_surface.get_rect()
                win_rate_rect.centerx = self.screen.get_width() // 2
                win_rate_rect.top = 10
                
                # 绘制背景框
                bg_rect = pygame.Rect(win_rate_rect.left - 10, win_rate_rect.top - 5, 
                                    win_rate_rect.width + 20, win_rate_rect.height + 10)
                pygame.draw.rect(self.screen, (240, 240, 240), bg_rect)
                pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 2)
                
                # 显示胜率文本
                self.screen.blit(win_rate_surface, win_rate_rect)

        pygame.display.flip()

    def show_help(self):
        """显示帮助信息"""
        help_text = [
            "按键说明:",
            "H - 显示此帮助信息",
            "N - 切换神经网络分析显示",
            "U - 撤销上一步",
            "鼠标左键 - 落子",
            "",
            "神经网络分析:",
            "绿色热度图显示:",
            "  - 概率前10的位置",
            "  - 概率大于55%的位置",
            "  - 概率越大，绿色越深",
            "灰色 - 已有棋子位置",
            "",
            "胜率显示:",
            "  - 显示在棋盘上方中央",
            "  - 黑方优势: 胜率>50%",
            "  - 白方优势: 胜率<50%",
            "  - 均势: 胜率≈50%"
        ]

        # 创建帮助窗口
        help_surface = pygame.Surface((400, 300))
        help_surface.fill((240, 240, 240))

        font = pygame.font.SysFont(None, 24)
        y_offset = 20
        for line in help_text:
            if line.strip():
                text = font.render(line, True, (0, 0, 0))
                help_surface.blit(text, (20, y_offset))
            y_offset += 30

        # 显示帮助窗口
        self.screen.blit(help_surface, (50, 50))
        pygame.display.flip()

        # 等待用户按键关闭
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    waiting = False
                    break
            pygame.time.wait(50)

    def play(self):
        current_player = 1
        while True:
            self.draw_board()
            if self.board.is_terminal():
                self.show_winner(self.board.get_winner().value.real)
                pygame.time.wait(1000)
                break

            if self.is_human_turn(current_player):
                res = self.get_human_move()
                move = res
            else:
                undone = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                if undone:
                    continue
                agent = self.agent_black if current_player == 1 else self.agent_white
                # 设置当前AI代理用于神经网络分析
                self.current_agent = agent
                move = agent.select_move(self.board.copy(), current_player)
                pygame.time.wait(self.delay)

            self.board.step(GomokuAction(move[0], move[1], current_player))
            current_player = -current_player


if __name__ == "__main__":
    # Example usage: human vs random agent
    model = PolicyValueNet(board_size=15)
    path = r"../check_dir/run6/model/strong_model_10.pth"
    # path = r"C:\Users\12700\Downloads\strong_model_50.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
    modelAgent = MCTSAgent(model)
    # game = PygameMatch(modelAgent, modelAgent)
    game = PygameMatch(None, modelAgent, board_size=15)
    game.play()
