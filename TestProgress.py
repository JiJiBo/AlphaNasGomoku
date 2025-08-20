import time
import multiprocessing as mp
import torch
import random
import numpy as np

from board.GomokuBoard import GomokuBoard, GomokuAction
from board.GomukuPlayer import PLAYER_BLACK, PLAYER_WHITE, PLAYER_EMPTY, Winner
from net.GomokuNet import PolicyValueNet
from mcts.MCTS_Agent import MCTS_Agent

mp.set_start_method('spawn', force=True)


def generate_random_safe_board(board_size=15, max_moves=50):
    """生成随机非终局棋盘"""
    board = GomokuBoard(size=board_size)
    moves = 0
    while moves < max_moves:
        available = board.available()
        if not available:
            break
        x, y = available[np.random.randint(len(available))]
        flag = PLAYER_BLACK if np.random.rand() < 0.5 else PLAYER_WHITE
        action = GomokuAction(x, y, flag)
        board.step(action)
        if board.get_winner() != Winner.EMPTY:
            board.board[x, y] = PLAYER_EMPTY
            board.history.pop()
            board.move_count -= 1
        else:
            moves += 1
    return board


def play_one_game(work_id, model_state_dict, board_size=15):
    """单局自对弈"""
    model = PolicyValueNet(board_size= board_size)
    model.load_state_dict(model_state_dict)
    model.eval()
    board = generate_random_safe_board(board_size)
    agent = MCTS_Agent(model, tau=1.0, c_puct=1.4)

    player = PLAYER_BLACK
    root = None
    while not board.is_terminal():
        info, cur_root = agent.run(board, player, is_train=True, cur_root=root)
        move, pi = info
        if cur_root.children[move] is not None:
            root = cur_root.children[move].child
        board.step(move)
        player = -player
    winner = board.get_winner()
    return winner.value.real


def worker(worker_id, model_state_dict, games_per_worker=5, board_size=15, result_queue=None):
    """每个进程跑固定局数"""
    start_time = time.time()
    results = []
    for i in range(games_per_worker):
        winner = play_one_game(worker_id, model_state_dict, board_size)
        results.append(winner)
    end_time = time.time()
    if result_queue is not None:
        result_queue.put((worker_id, results, end_time - start_time))


def run_test(num_processes=4, total_games=20, board_size=15):
    """多进程自对弈测试"""
    games_per_worker = total_games // num_processes
    extra = total_games % num_processes

    # 初始化模型（随机初始化即可）
    model = PolicyValueNet(board_size)
    model_state_dict = model.state_dict()

    result_queue = mp.Queue()
    processes = []

    start_time = time.time()
    for i in range(num_processes):
        n_games = games_per_worker + (1 if i < extra else 0)
        p = mp.Process(target=worker, args=(i, model_state_dict, n_games, board_size, result_queue))
        p.start()
        processes.append(p)

    total_results = []
    while any(p.is_alive() for p in processes) or not result_queue.empty():
        try:
            worker_id, results, duration = result_queue.get(timeout=5)
            print(f"[Worker {worker_id}] 完成 {len(results)} 局，自耗时 {duration:.2f}s")
            total_results.extend(results)
        except:
            pass

    for p in processes:
        p.join()

    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f}s")
    print(f"总局数: {len(total_results)}")
    wins_black = sum(1 for w in total_results if w == PLAYER_BLACK)
    wins_white = sum(1 for w in total_results if w == PLAYER_WHITE)
    draws = sum(1 for w in total_results if w == 0)
    print(f"黑棋胜: {wins_black}, 白棋胜: {wins_white}, 平局: {draws}")
    print(f"吞吐量: {len(total_results) / (end_time - start_time):.2f} 局/s")


if __name__ == "__main__":
    for n in [1, 2, 4, 8, 12, 16, 20]:
        print(f"\n=== 使用 {n} 个进程 ===")
        run_test(num_processes=n, total_games=20)
