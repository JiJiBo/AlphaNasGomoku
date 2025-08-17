import os
import random
import time
import torch.nn.functional as F
import numpy as np
import torch
from numpy import mean
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from board.GomokuBoard import GomokuBoard
from board.GomukuPlayer import PLAYER_BLACK, PLAYER_WHITE
from datasets.DataSets import Weighted_Dataset
from mcts.MCTS_Agent import MCTS_Agent
from net.GomokuNet import PolicyValueNet
import multiprocessing
from functools import partial

def generate_selfplay_data(strong_model, weak_model, num_games, board_size):
    # 使用多进程并行生成游戏

    strong_model_state_dict = strong_model.state_dict()
    weak_model_state_dict = weak_model.state_dict()
    with multiprocessing.get_context('spawn').Pool(
            processes=22
    ) as pool:
        func = partial(
            gen_a_episode_data,
            strong_model_state_dict=strong_model_state_dict,
            weak_model_state_dict=weak_model_state_dict,
            board_size=board_size,
        )
        results = list(tqdm(
            pool.imap(func, range(100)),
            total=100,
            desc="Generating games"
        ))

    # 整合结果
    boards, policies, values, weights = [], [], [], []
    for game_boards, game_policies, game_values, game_weights in results:
        boards.extend(game_boards)
        policies.extend(game_policies)
        values.extend(game_values)
        weights.extend(game_weights)

    boards = torch.stack(boards)
    policies = torch.stack(policies)
    values = torch.FloatTensor(values)
    weights = torch.FloatTensor(weights)

    # 数据增强：旋转和翻转
    print("len_boards_raw = ", len(boards))
    boards, policies, values, weights = augment_data(boards, policies, values, weights)

    return boards, policies, values, weights


def gen_a_episode_data(_, strong_model_state_dict, weak_model_state_dict, board_size):
    strong_model = PolicyValueNet(board_size=board_size)
    strong_model.load_state_dict(strong_model_state_dict)
    strong_model.eval()  # 设置为评估模式
    weak_model = PolicyValueNet(board_size=board_size)
    weak_model.load_state_dict(weak_model_state_dict)
    weak_model.eval()  # 设置为评估模式
    sum_boards, sum_policies, sum_values, sum_weights = [], [], [], []
    strong_agent = MCTS_Agent(strong_model)
    weak_agent = MCTS_Agent(weak_model)
    root_bord = GomokuBoard(board_size)
    player = PLAYER_BLACK
    while not root_bord.is_terminal():
        if player == PLAYER_BLACK:
            best_move, pi = strong_agent.run(root_bord, player, is_train=True)
        elif player == PLAYER_WHITE:
            best_move, pi = weak_agent.run(root_bord, player, is_train=True)
        else:
            raise RuntimeError("错误的执棋者")
        root_bord.step(best_move)
        player = -player
    boards, policies, values, weights = strong_agent.get_train_data()
    sum_boards.append(boards)
    sum_policies.append(policies)
    sum_values.append(values)
    sum_weights.append(weights)
    return sum_boards, sum_policies, sum_values, sum_weights


def train_model(model, train_loader, val_loader, writter, scheduler, optimizer):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    value_criterion = nn.MSELoss(reduction='none')
    policy_criterion = nn.KLDivLoss(reduction='none')
    val_value_criterion = nn.MSELoss()
    val_policy_criterion = nn.KLDivLoss(reduction='batchmean')
    train_losses = []
    val_losses = []

    print(f"开始训练，使用设备: {device}")

    for epoch in range(3):
        # 训练阶段
        model.train()
        train_value_loss, train_policy_loss = 0, 0

        for batch_boards, batch_policies, batch_values, batch_weights in tqdm(train_loader,
                                                                              desc=f'Epoch {epoch + 1} '):
            batch_boards = batch_boards.to(device)
            batch_policies = batch_policies.to(device).view(batch_policies.size(0), -1)
            batch_values = batch_values.to(device).unsqueeze(1)  # 添加维度匹配
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad()

            pred_values, pred_policies = model(batch_boards)

            # 计算损失
            value_loss = value_criterion(pred_values, batch_values).squeeze(1)
            policy_loss = policy_criterion(F.log_softmax(pred_policies, dim=1),
                                           batch_policies.view(-1, batch_policies.size(-1))).sum(dim=1, keepdim=True)
            # print(value_loss.shape)
            # print(policy_loss.shape)
            # print(pred_values.shape)
            # print(pred_policies.shape)

            weighted_value_loss = (value_loss * batch_weights).mean()
            weighted_policy_loss = (policy_loss * batch_weights).mean()

            loss = 2 * weighted_value_loss + weighted_policy_loss

            loss.backward()
            optimizer.step()

            train_value_loss += weighted_value_loss.item()
            train_policy_loss += weighted_policy_loss.item()

        model.eval()
        val_value_loss, val_policy_loss = 0, 0
        with torch.no_grad():
            for boards, policies, values, weights in val_loader:
                boards = boards.to(device)
                policies = policies.to(device).view(policies.size(0), -1)
                values = values.to(device).unsqueeze(1)

                pred_values, pred_policies = model(boards)
                val_value_loss += val_value_criterion(pred_values, values).item()
                val_policy_loss += val_policy_criterion(
                    F.log_softmax(pred_policies, dim=1),
                    policies.view(-1, policies.size(-1))
                ).item()

        # 计算平均损失
        avg_train_value = train_value_loss / len(train_loader)
        avg_train_policy = train_policy_loss / len(train_loader)
        avg_val_value = val_value_loss / len(val_loader)
        avg_val_policy = val_policy_loss / len(val_loader)

        print("Train: ", train_value_loss, train_policy_loss)
        print("Val: ", val_value_loss, val_policy_loss)
        scheduler.step()
        train_losses.append(avg_train_value + avg_train_policy)
        val_losses.append(avg_val_value + avg_val_policy)

    return train_losses, val_losses


def trian():
    board_size = 19
    batch_size = 256
    epochs = 120
    train_ratio = 0.9
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs('./check_dir', exist_ok=True)
    run_id = len(os.listdir('./check_dir'))
    checkpoints_path = f"./check_dir/run{run_id}"
    os.makedirs(checkpoints_path, exist_ok=True)
    log_dir = os.path.join("/root/tf-logs/", "log")
    writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S")))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strong_model = PolicyValueNet(board_size=board_size).to(device)
    # resume_path = "./check_dir/run36/model_step20.pth"
    # if os.path.exists(resume_path):
    #     strong_model.load_state_dict(torch.load(resume_path, map_location=device))
    weak_model = PolicyValueNet(board_size=board_size).to(device)
    optimizer = torch.optim.Adam(strong_model.parameters(), lr=0.2)
    milestones = [30, 60, 90]
    gamma = 0.1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    eps = tqdm(range(epochs))
    for epoch in eps:
        strong_model.train()
        weak_model.train()
        sum_boards, sum_policies, sum_values, sum_weights = generate_selfplay_data(strong_model, weak_model, 22,
                                                                                   board_size)
        # 划分训练集和验证集
        num_train = int(len(sum_boards) * train_ratio)
        train_boards = sum_boards[:num_train]
        train_policies = sum_policies[:num_train]
        train_values = sum_values[:num_train]
        train_weights = sum_weights[:num_train]

        val_boards = sum_boards[num_train:]
        val_policies = sum_policies[num_train:]
        val_values = sum_values[num_train:]
        val_weights = sum_weights[num_train:]

        # 创建数据集和数据加载器
        train_dataset = Weighted_Dataset(train_boards, train_policies, train_values, train_weights)
        val_dataset = Weighted_Dataset(val_boards, val_policies, val_values, val_weights)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 训练模型
        train_losses, val_losses = train_model(strong_model, train_loader, val_loader, writer, scheduler, optimizer)
        writer.add_scalar('loss/train_losses', mean(train_losses), epoch)
        writer.add_scalar('loss/val_losses', mean(val_losses), epoch)


if __name__ == '__main__':
    trian()
