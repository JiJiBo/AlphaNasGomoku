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
import multiprocessing as mp

from board.GomokuBoard import GomokuBoard
from board.GomukuPlayer import PLAYER_BLACK, PLAYER_WHITE
from datasets.DataSets import Weighted_Dataset
from mcts.MCTS_Agent import MCTS_Agent
from net.GomokuNet import PolicyValueNet

mp.set_start_method('spawn', force=True)


def generate_selfplay_data(epoch, strong_model, weak_model, num_games, board_size, max_games_per_worker=5, ):
    device = torch.device('cpu')
    strong_model_state_dict = strong_model.to(device).state_dict()
    weak_model_state_dict = weak_model.to(device).state_dict()

    data_queue = mp.Queue()
    stop_event = mp.Event()

    workers = []
    for i in range(num_games):
        p = mp.Process(target=gen_a_episode_data,
                       args=(i, epoch, strong_model_state_dict, weak_model_state_dict, board_size, data_queue,
                             stop_event,
                             max_games_per_worker))
        p.start()
        workers.append(p)

    batch = []
    total_strong_wins = 0
    total_weak_wins = 0
    total_draws = 0

    while any(p.is_alive() for p in workers) or not data_queue.empty():
        try:
            item = data_queue.get(timeout=5)
            if item[0] == 'results':
                total_strong_wins += item[1]
                total_weak_wins += item[2]
                total_draws += item[3]
            else:
                batch.append(item)
        except Exception:
            continue

    stop_event.set()
    for p in workers:
        p.join()

    boards, policies, values, weights = [], [], [], []
    for b, p, v, w in batch:
        boards.append(b)
        policies.append(p)
        values.append(v)
        weights.append(w)

    boards = torch.stack(boards)
    policies = torch.stack(policies)
    values = torch.FloatTensor(values)
    weights = torch.FloatTensor(weights)

    total_games = total_strong_wins + total_weak_wins + total_draws
    if total_games > 0:
        strong_win_rate = total_strong_wins / total_games
        weak_win_rate = total_weak_wins / total_games
        draw_rate = total_draws / total_games
        print(f"对战统计 - 强模型胜率: {strong_win_rate:.2%}, 弱模型胜率: {weak_win_rate:.2%}, 平局率: {draw_rate:.2%}")

    print(f"生成完毕，样本总数: {len(boards)}")
    return boards, policies, values, weights, total_strong_wins, total_weak_wins, total_draws


def get_c_puct(epoch, c_start=4.5, c_end=1.0, max_epoch=100):
    """
    前 max_epoch 个 epoch 线性从 c_start 减到 c_end
    之后保持不变
    """
    if epoch <= max_epoch:
        return c_start + (c_end - c_start) * (epoch / max_epoch)
    else:
        return c_end


def get_tau(epoch: int, mode: str = 'linear',
            tau_start: float = 1.0, tau_min: float = 0.01,
            epoch_start: int = 0, epoch_end: int = 100, decay_rate: float = 0.05) -> float:
    """
    根据当前 epoch 返回温度 tau

    参数:
        epoch       : 当前训练轮次
        mode        : 'linear' 或 'exp' 指定衰减模式
        tau_start   : 初始温度
        tau_min     : 最低温度
        epoch_start : 开始衰减的 epoch
        epoch_end   : 结束 epoch（仅线性衰减用）
        decay_rate  : 衰减速率（仅指数衰减用）

    返回:
        tau : 当前温度
    """
    if mode == 'linear':
        if epoch <= epoch_start:
            return tau_start
        elif epoch >= epoch_end:
            return tau_min
        else:
            # 线性衰减
            return tau_start - (tau_start - tau_min) * (epoch - epoch_start) / (epoch_end - epoch_start)
    elif mode == 'exp':
        # 指数衰减
        tau = tau_min + (tau_start - tau_min) * (2.71828 ** (-decay_rate * epoch))
        return max(tau, tau_min)
    else:
        raise ValueError("mode must be 'linear' or 'exp'")


def gen_a_episode_data(work_id, epoch, strong_model_state_dict, weak_model_state_dict, board_size, data_queue,
                       stop_event,
                       max_games):
    torch.set_num_threads(min(mp.cpu_count() // 4, 4))

    strong_model = PolicyValueNet(board_size=board_size)
    strong_model.load_state_dict(strong_model_state_dict)
    strong_model.eval()

    weak_model = PolicyValueNet(board_size=board_size)
    weak_model.load_state_dict(weak_model_state_dict)
    weak_model.eval()

    strong_wins = 0
    weak_wins = 0
    draws = 0
    tau = get_tau(epoch)
    c_puct = get_c_puct(epoch)
    for _ in tqdm(range(max_games), desc=f"work: {work_id} - epoch: {epoch}"):
        if stop_event.is_set():
            break
        board = GomokuBoard(board_size)

        strong_agent = MCTS_Agent(strong_model, tau=tau, c_puct=c_puct, )
        weak_agent = MCTS_Agent(weak_model, tau=tau, c_puct=c_puct, )
        # 随机决定哪个模型用白棋
        if random.random() < 0.5:
            white_agent, black_agent = strong_agent, weak_agent
            strong_is_white = True
        else:
            white_agent, black_agent = weak_agent, strong_agent
            strong_is_white = False

        player = PLAYER_BLACK
        # print(f"{work_id} 第一个打手 ", "黑棋" if player == 1 else "白棋", "强势者 是 ", "白棋" if  strong_is_white else "黑棋")
        while not board.is_terminal():
            if player == PLAYER_WHITE:
                move, pi = white_agent.run(board, player, is_train=True)
            else:
                move, pi = black_agent.run(board, player, is_train=True)
            board.step(move)
            player = -player

        # 统计胜负
        winner = board.get_winner().value.real
        if winner == PLAYER_WHITE:
            print(f"白棋 ✌️赢了!", "强势者 是 ", "白棋" if strong_is_white else "黑棋")
        if winner == PLAYER_WHITE and strong_is_white:
            strong_wins += 1
            # print(f"强加一 现在强: {strong_wins} 现在弱:{weak_wins}")
        elif winner == PLAYER_BLACK and not strong_is_white:
            strong_wins += 1
            # print(f"强加一 现在强: {strong_wins} 现在弱:{weak_wins}")
        elif winner == 0:
            draws += 1
            # print(f"平加一 现在强: {strong_wins} 现在弱:{weak_wins}")
        else:
            weak_wins += 1
            # print(f"弱加一 现在强: {strong_wins} 现在弱:{weak_wins}")
        # wStr = "黑棋" if winner == 1 else "白棋"
        # print(f"work {work_id}  赢家是{wStr}")
        boards, policies, values, weights = strong_agent.get_train_data()
        for b, p, v, w in zip(boards, policies, values, weights):
            try:
                data_queue.put((b, p.reshape(-1), v, w), timeout=1)
            except mp.queues.Full:
                continue

    # 将胜负结果也放入队列
    try:
        data_queue.put(('results', strong_wins, weak_wins, draws), timeout=1)
    except mp.queues.Full:
        pass


def train_model(model, train_loader, val_loader, writer, scheduler, optimizer):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    value_criterion = nn.MSELoss(reduction='none')
    val_value_criterion = nn.MSELoss()
    train_losses = []
    val_losses = []

    print(f"开始训练，使用设备: {device}")
    print(f"[训练开始] 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    for epoch in range(3):  # 外层train已经控制epoch，所以这里只跑一次
        model.train()
        train_value_loss, train_policy_loss = [], []

        for batch_boards, batch_policies, batch_values, batch_weights in tqdm(train_loader,
                                                                              desc=f'Train epoch'):
            batch_boards = batch_boards.to(device)
            batch_policies = batch_policies.to(device).view(batch_policies.size(0), -1)
            # 保证 target 策略是概率分布
            batch_policies = F.softmax(batch_policies, dim=1)
            batch_values = batch_values.to(device).unsqueeze(1)
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad()
            # 得到 先验概率 pred_policies 和 胜率 pred_values
            # pred_policies 没有 soft Max
            pred_policies, pred_values = model(batch_boards)

            # value 损失
            value_loss = value_criterion(pred_values, batch_values).squeeze(1)

            # policy 损失（交叉熵形式）
            log_probs = F.log_softmax(pred_policies, dim=1)
            policy_loss = -(batch_policies * log_probs).sum(dim=1)

            # 加权平均
            weighted_value_loss = (value_loss * batch_weights).mean()
            weighted_policy_loss = (policy_loss * batch_weights).mean()

            # -------- 动态平衡权重 --------
            with torch.no_grad():
                std_v = value_loss.std().item() + 1e-6
                std_p = policy_loss.std().item() + 1e-6
                alpha = std_p / (std_v + std_p)
                beta = std_v / (std_v + std_p)

            loss = alpha * weighted_value_loss + beta * weighted_policy_loss

            loss.backward()
            optimizer.step()

            train_value_loss.append(weighted_value_loss.item())
            train_policy_loss.append(weighted_policy_loss.item())

        scheduler.step()
        model.eval()
        val_value_loss, val_policy_loss = 0, 0
        with torch.no_grad():
            for boards, policies, values, weights in val_loader:
                boards = boards.to(device)
                policies = policies.to(device).view(policies.size(0), -1)
                policies = F.softmax(policies, dim=1)  # 保证验证时也是概率分布
                values = values.to(device).unsqueeze(1)

                pred_policies, pred_values = model(boards)

                # value loss
                val_value_loss += val_value_criterion(pred_values, values).item()

                # policy loss (手动交叉熵，保持和训练一致)
                log_probs = F.log_softmax(pred_policies, dim=1)
                val_policy_loss += (-(policies * log_probs).sum(dim=1)).mean().item()

        avg_train_value = np.mean(train_value_loss)
        avg_train_policy = np.mean(train_policy_loss)
        avg_val_value = val_value_loss / len(val_loader)
        avg_val_policy = val_policy_loss / len(val_loader)

        print("Train: ", avg_train_value, avg_train_policy)
        print("Val: ", avg_val_value, avg_val_policy)

        train_losses.append(avg_train_value + avg_train_policy)
        val_losses.append(avg_val_value + avg_val_policy)

    return train_losses, val_losses


def train():
    board_size = 15
    batch_size = 256
    epochs = 200
    train_ratio = 0.9
    seed = 42
    win_rate_threshold = 0.55  # 胜率阈值
    window_size = 20

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
    weak_model = PolicyValueNet(board_size=board_size).to(device)

    # 加载预训练模型
    resume_Dir = ""
    if os.path.exists(resume_Dir):
        print(f"加载预训练模型: {resume_Dir}")
        strong_model.load_state_dict(torch.load(resume_Dir, map_location=device))
        weak_model.load_state_dict(torch.load(resume_Dir, map_location=device))
    else:
        print("未找到预训练模型，从头开始训练")
    optimizer = torch.optim.Adam(strong_model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 添加胜率跟踪
    recent_results = []  # 存储最近的胜负结果 (1=强模型胜, -1=弱模型胜, 0=平局)
    last_sync_epoch = 0
    eps = tqdm(range(epochs))
    for epoch in eps:
        strong_model.train()
        weak_model.train()
        sum_boards, sum_policies, sum_values, sum_weights, strong_wins, weak_wins, draws = generate_selfplay_data(epoch,
                                                                                                                  strong_model,
                                                                                                                  weak_model,
                                                                                                                  20,
                                                                                                                  board_size)

        # 更新最近结果
        for _ in range(strong_wins):
            recent_results.append(1)
        for _ in range(weak_wins):
            recent_results.append(-1)
        for _ in range(draws):
            recent_results.append(0)

        # 保持只记录最近的window_size局
        recent_results = recent_results[-window_size:]

        # 计算最近胜率
        if len(recent_results) >= window_size:
            recent_strong_wins = sum(1 for r in recent_results if r == 1)
            recent_win_rate = recent_strong_wins / window_size

            # 如果胜率达到阈值，更新弱模型
            if recent_win_rate >= win_rate_threshold and epoch - last_sync_epoch >= 20:
                last_sync_epoch = epoch
                print(f"强模型最近{window_size}局胜率{recent_win_rate:.2%}达到阈值{win_rate_threshold:.0%}，更新弱模型")
                weak_model.load_state_dict(strong_model.state_dict())
                recent_results = []  # 重置胜率统计
            else:
                print(f"强模型最近{window_size}局胜率{recent_win_rate:.2%}")
        total_strong_wins = sum(1 for i in recent_results if i == 1)

        writer.add_scalar('strong_wins', total_strong_wins / len(recent_results), epoch)
        # 划分训练集和验证集
        # 打乱数据
        num_samples = len(sum_boards)
        indices = torch.randperm(num_samples)

        num_train = int(num_samples * train_ratio)
        train_idx = indices[:num_train]
        val_idx = indices[num_train:]

        # 训练集
        train_boards = sum_boards[train_idx]
        train_policies = sum_policies[train_idx]
        train_values = sum_values[train_idx]
        train_weights = sum_weights[train_idx]

        # 验证集
        val_boards = sum_boards[val_idx]
        val_policies = sum_policies[val_idx]
        val_values = sum_values[val_idx]
        val_weights = sum_weights[val_idx]

        # 创建数据集和数据加载器
        train_dataset = Weighted_Dataset(train_boards, train_policies, train_values, train_weights)
        val_dataset = Weighted_Dataset(val_boards, val_policies, val_values, val_weights)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 训练模型
        train_losses, val_losses = train_model(strong_model, train_loader, val_loader, writer, scheduler, optimizer)
        writer.add_scalar('loss/train_losses', mean(train_losses), epoch)
        writer.add_scalar('loss/val_losses', mean(val_losses), epoch)

        if epoch % 10 == 0:
            model_dir = os.path.join(checkpoints_path, "model")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(strong_model.state_dict(), os.path.join(model_dir, f"strong_model_{epoch}.pth"))

    torch.save(strong_model.state_dict(), os.path.join(checkpoints_path, "strong_model.pth"))


if __name__ == '__main__':
    print(f"[训练开始] 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    train()
    print(f"[训练结束] 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
