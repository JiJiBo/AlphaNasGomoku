import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from numpy import mean
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from board.GomokuBoard import GomokuBoard, GomokuAction
from board.GomukuPlayer import PLAYER_BLACK, PLAYER_WHITE, Winner, PLAYER_EMPTY, KL_TARG
from datasets.DataSets import Weighted_Dataset
from mcts.MCTS_Agent import MCTS_Agent
from net.GomokuNet import PolicyValueNet

mp.set_start_method("spawn", force=True)


def generate_random_safe_board(board_size=6, max_moves=None, max_attempts_per_move=3):
    """
    生成随机非终局棋盘（安全棋盘）
    :param board_size: 棋盘大小
    :param max_moves: 最大落子步数
    :param max_attempts_per_move: 每一步最大尝试次数，避免落子导致胜利
    :return: GomokuBoard 对象
    """
    board = GomokuBoard(size=board_size)
    if max_moves is None:
        max_moves = random.randint(2, random.randint(2, board_size * board_size // 4))

    moves = 0
    while moves < max_moves:
        available = board.available()
        if not available:
            break

        success = False
        attempts = 0
        while not success and attempts < max_attempts_per_move:
            x, y = available[np.random.randint(len(available))]
            flag = PLAYER_BLACK if np.random.rand() < 0.5 else PLAYER_WHITE
            action = GomokuAction(x, y, flag)

            board.step(action)

            # 检查是否产生赢家
            if board.get_winner() == Winner.EMPTY:
                success = True
            else:
                # 产生赢家则撤销
                board.board[x, y] = PLAYER_EMPTY
                board.history.pop()
                board.move_count -= 1
                attempts += 1

        if success:
            moves += 1
        else:
            # 当前步尝试多次仍无法安全落子，停止生成
            break

    return board


def generate_selfplay_data(
    epoch,
    strong_model,
    weak_model,
    num_games,
    board_size,
    max_games_per_worker=7,
    nc=400,
):
    device = torch.device("cpu")
    strong_model_state_dict = strong_model.to(device).state_dict()
    weak_model_state_dict = weak_model.to(device).state_dict()

    data_queue = mp.Queue()
    stop_event = mp.Event()

    workers = []
    for i in range(num_games):
        p = mp.Process(
            target=gen_a_episode_data,
            args=(
                i,
                epoch,
                strong_model_state_dict,
                weak_model_state_dict,
                board_size,
                data_queue,
                stop_event,
                max_games_per_worker,
                nc,
            ),
        )
        p.start()
        workers.append(p)

    batch = []
    total_strong_wins = 0
    total_weak_wins = 0
    total_draws = 0

    while any(p.is_alive() for p in workers) or not data_queue.empty():
        try:
            item = data_queue.get(timeout=5)
            if item[0] == "results":
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
        print(
            f"对战统计 - 强模型胜率: {strong_win_rate:.2%}, 弱模型胜率: {weak_win_rate:.2%}, 平局率: {draw_rate:.2%}"
        )

    print(f"生成完毕，样本总数: {len(boards)}")
    return (
        boards,
        policies,
        values,
        weights,
        total_strong_wins,
        total_weak_wins,
        total_draws,
    )


def get_c_puct(epoch, c_start=1.4, c_end=0.5, max_epoch=20):
    """
    前 max_epoch 个 epoch 线性从 c_start 减到 c_end
    之后保持不变
    """
    if epoch <= max_epoch:
        return c_start + (c_end - c_start) * (epoch / max_epoch)
    else:
        return c_end


def get_tau(
    epoch: int,
    mode: str = "linear",
    tau_start: float = 1.0,
    tau_min: float = 0.01,
    epoch_start: int = 0,
    epoch_end: int = 20,
    decay_rate: float = 0.05,
) -> float:
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
    if mode == "linear":
        if epoch <= epoch_start:
            return tau_start
        elif epoch >= epoch_end:
            return tau_min
        else:
            # 线性衰减
            return tau_start - (tau_start - tau_min) * (epoch - epoch_start) / (
                epoch_end - epoch_start
            )
    elif mode == "exp":
        # 指数衰减
        tau = tau_min + (tau_start - tau_min) * (2.71828 ** (-decay_rate * epoch))
        return max(tau, tau_min)
    else:
        raise ValueError("mode must be 'linear' or 'exp'")


def gen_a_episode_data(
    work_id,
    epoch,
    strong_model_state_dict,
    weak_model_state_dict,
    board_size,
    data_queue,
    stop_event,
    max_games,
    nc,
):
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
    # for _ in range(max_games):
    for _ in tqdm(range(max_games), desc=f"work: {work_id} - epoch: {epoch}"):
        if stop_event.is_set():
            break
        # board = generate_random_safe_board(board_size)
        board = GomokuBoard(board_size)
        strong_agent = MCTS_Agent(
            strong_model,
            tau=tau,
            c_puct=c_puct,
        )
        weak_agent = MCTS_Agent(
            weak_model,
            tau=tau,
            c_puct=c_puct,
        )
        # 随机决定哪个模型用白棋
        # 随机决定哪个模型用白棋
        # if random.random() < 0.5:
        black_agent, white_agent = strong_agent, weak_agent
        strong_is_white = False
        # else:
        #     black_agent, white_agent = weak_agent, strong_agent
        #     strong_is_white = True
        player = PLAYER_BLACK
        root = None
        # print(f"{work_id} 第一个打手 ", "黑棋" if player == 1 else "白棋", "强势者 是 ", "白棋" if  strong_is_white else "黑棋")
        while not board.is_terminal():
            if player == PLAYER_WHITE:
                info, cur_root = white_agent.run(
                    board, player, is_train=True, cur_root=root, number_samples=nc
                )
                move, pi = info
            else:
                info, cur_root = black_agent.run(
                    board, player, is_train=True, cur_root=root, number_samples=nc
                )
                move, pi = info
            if cur_root.children[move] is not None:
                edge = cur_root.children[move]
                root = edge.child
            board.step(move)
            player = -player

        # 统计胜负
        winner = board.get_winner().value.real
        # if winner == PLAYER_WHITE:
        #     print(f"白棋 ✌️赢了!", "强势者 是 ", "白棋" if strong_is_white else "黑棋")
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
        # 在一局对局结束后

        # 获取训练数据（使用最终赢家）
        for agent in [strong_agent]:
            boards, policies, values, weights = agent.get_train_data(winner)
            for b, p, v, w in zip(boards, policies, values, weights):
                try:
                    data_queue.put((b, p.reshape(-1), v, w), timeout=2)
                except mp.queues.Full:
                    pass  # 队列满了就丢弃或可重试

    # 将胜负结果也放入队列
    try:
        data_queue.put(("results", strong_wins, weak_wins, draws), timeout=1)
    except mp.queues.Full:
        pass


def train_model(model, train_loader, val_loader, writer, epochs, lr_multiplier):
    """
    标准 AlphaZero 风格训练
    """
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("当前学习率是", lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    value_criterion = nn.MSELoss(reduction="none")  # 可加权
    val_value_criterion = nn.MSELoss()  # 验证集直接 MSE

    train_losses, val_losses = [], []

    for epoch in range(5):  # 外层已经控制大循环，这里小循环即可
        model.train()
        train_value_loss, train_policy_loss = [], []

        for batch_boards, batch_policies, batch_values, batch_weights in tqdm(
            train_loader
        ):
            batch_boards = batch_boards.to(device)
            batch_policies = batch_policies.to(device).view(
                batch_policies.size(0), model.board_size, model.board_size
            )
            batch_values = batch_values.to(device).unsqueeze(1)
            batch_weights = batch_weights.to(device).unsqueeze(1)
            with torch.no_grad():
                pred_policies_old, pred_values_old = model(batch_boards)
            optimizer.zero_grad()
            # 网络输出
            pred_policies, pred_values = model(batch_boards)

            print("lr multiplier:", lr_multiplier)
            print("lr lr * lr_multiplier:", lr * lr_multiplier)
            for params in optimizer.param_groups:
                # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9
                params["lr"] = lr * lr_multiplier
            # ---- value loss ----
            value_loss = value_criterion(pred_values, batch_values).squeeze(1)
            weighted_value_loss = (value_loss * batch_weights).mean()

            # ---- policy loss ----
            # 注意: batch_policies 已经是访问次数归一化的概率分布，不需要 softmax
            log_probs = F.log_softmax(pred_policies, dim=1)

            policy_loss = -(batch_policies * log_probs).sum(dim=1)
            weighted_policy_loss = (policy_loss * batch_weights).mean()

            loss = weighted_value_loss + weighted_policy_loss

            loss.backward()
            optimizer.step()

            train_value_loss.append(weighted_value_loss.item())
            train_policy_loss.append(weighted_policy_loss.item())
            with torch.no_grad():
                pred_policies_new, pred_values_new = model(batch_boards)
            pp_old = pred_policies_old
            pp_old = torch.exp(pp_old)
            pred_policies_new = torch.exp(pred_policies_new)
            # 计算KL散度
            KL_LOSS = torch.mean(
                torch.sum(
                    pp_old
                    * (
                        torch.log(pp_old + 1e-10) - torch.log(pred_policies_new + 1e-10)
                    ),
                    dim=1,
                )
            )
            print("KL:", KL_LOSS.item())
            if KL_LOSS > KL_TARG * 4:  # 如果KL散度很差，则提前终止
                print("KL散度很差，提前终止")
                break
            if KL_LOSS > KL_TARG * 2 and lr_multiplier > 0.1:
                lr_multiplier /= 1.5
            elif KL_LOSS < KL_TARG / 2 and lr_multiplier < 10:
                lr_multiplier *= 1.5
        # ---- 验证 ----
        model.eval()
        val_value_loss, val_policy_loss = 0, 0
        with torch.no_grad():
            for boards, policies, values, weights in val_loader:
                boards = boards.to(device)
                policies = policies.to(device).view(
                    policies.size(0), model.board_size, model.board_size
                )
                values = values.to(device).unsqueeze(1)

                pred_policies, pred_values = model(boards)

                # value
                val_value_loss += val_value_criterion(pred_values, values).mean().item()

                # policy
                log_probs = F.log_softmax(pred_policies, dim=1)
                val_policy_loss += (-(policies * log_probs).sum(dim=1)).mean().item()

        avg_train_value = np.mean(train_value_loss)
        avg_train_policy = np.mean(train_policy_loss)
        avg_val_value = val_value_loss / len(val_loader)
        avg_val_policy = val_policy_loss / len(val_loader)

        print(f"Train - Value: {avg_train_value:.4f}, Policy: {avg_train_policy:.4f}")
        print(f"Val   - Value: {avg_val_value:.4f}, Policy: {avg_val_policy:.4f}")

        train_losses.append(avg_train_value + avg_train_policy)
        val_losses.append(avg_val_value + avg_val_policy)

    return train_losses, val_losses, lr_multiplier


def train():
    board_size = 6
    batch_size = 8
    epochs = 3000
    train_ratio = 0.9
    seed = 42
    win_rate_threshold = 0  # 胜率阈值
    window_size = 30
    nc = 1200
    lr_multiplier = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs("./check_dir", exist_ok=True)
    run_id = len(os.listdir("./check_dir"))
    checkpoints_path = f"./check_dir/run{run_id}"
    os.makedirs(checkpoints_path, exist_ok=True)
    log_dir = os.path.join(checkpoints_path, "log")
    writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strong_model = PolicyValueNet(board_size=board_size).to(device)
    weak_model = PolicyValueNet(board_size=board_size).to(device)

    # 加载预训练模型
    resume_Dir = "./check_dir/run6/model/strong_model_10.pth"
    if os.path.exists(resume_Dir):
        print(f"加载预训练模型: {resume_Dir}")
        strong_model.load_state_dict(torch.load(resume_Dir, map_location=device))
        weak_model.load_state_dict(torch.load(resume_Dir, map_location=device))
    else:
        print("未找到预训练模型，从头开始训练")

    # 添加胜率跟踪
    recent_results = []  # 存储最近的胜负结果 (1=强模型胜, -1=弱模型胜, 0=平局)
    last_sync_epoch = 0
    eps = tqdm(range(epochs))
    for epoch in eps:
        strong_model.train()
        weak_model.train()
        (
            sum_boards,
            sum_policies,
            sum_values,
            sum_weights,
            strong_wins,
            weak_wins,
            draws,
        ) = generate_selfplay_data(
            epoch, strong_model, weak_model, 2, board_size, nc=nc
        )

        # 更新最近结果
        for _ in range(strong_wins):
            recent_results.append(1)
        for _ in range(weak_wins):
            recent_results.append(-1)
        for _ in range(draws):
            recent_results.append(0)

        # 保持只记录最近的window_size局
        recent_results = recent_results[-window_size:]

        recent_strong_wins = recent_results.count(1)
        recent_win_rate = recent_strong_wins / len(recent_results)
        # 如果胜率达到阈值，更新弱模型
        if recent_win_rate >= win_rate_threshold and epoch - last_sync_epoch >= 20:
            last_sync_epoch = epoch
            print(
                f"强模型最近{window_size}局胜率{recent_win_rate:.2%}达到阈值{win_rate_threshold:.0%}，更新弱模型"
            )
            weak_model.load_state_dict(strong_model.state_dict())
        else:
            print(f"强模型最近{window_size}局胜率{recent_win_rate:.2%}")
        total_strong_wins = recent_results.count(1)
        print("recent_results 的 长度 ", len(recent_results))
        print("数据 的 长度 ", len(sum_boards))
        writer.add_scalar("strong_wins", total_strong_wins / len(recent_results), epoch)
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
        train_dataset = Weighted_Dataset(
            train_boards, train_policies, train_values, train_weights
        )
        val_dataset = Weighted_Dataset(
            val_boards, val_policies, val_values, val_weights
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 训练模型
        train_losses, val_losses, lr_multiplier = train_model(
            strong_model, train_loader, val_loader, writer, epochs, lr_multiplier
        )
        writer.add_scalar("loss/train_losses", mean(train_losses), epoch)
        writer.add_scalar("loss/val_losses", mean(val_losses), epoch)

        if epoch % 1 == 0:
            model_dir = os.path.join(checkpoints_path, "model")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(
                strong_model.state_dict(),
                os.path.join(model_dir, f"strong_model_{epoch}.pth"),
            )

    torch.save(
        strong_model.state_dict(), os.path.join(checkpoints_path, "strong_model.pth")
    )


if __name__ == "__main__":
    print(
        f"[训练开始] 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )
    train()
    print(
        f"[训练结束] 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )
