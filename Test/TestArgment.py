import torch
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# augment_data 函数
# ---------------------------
def augment_data(boards, policies, values, weights):
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


# ---------------------------
# 可视化函数
# ---------------------------
def visualize_augmented_boards(original_board, augmented_boards):
    num_aug = augmented_boards.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(num_aug):
        # 用 sum 叠加所有通道
        board_img = augmented_boards[i].sum(dim=0).numpy()
        axes[i].imshow(board_img, cmap='gray', origin='upper')
        axes[i].set_title(f'Aug {i + 1}')
        axes[i].axis('off')

    # 原始棋盘
    orig_img = original_board.sum(dim=0).numpy()
    plt.figure()
    plt.imshow(orig_img, cmap='gray', origin='upper')
    plt.title("Original Board")
    plt.axis('off')
    plt.show()


# ---------------------------
# 测试脚本
# ---------------------------
if __name__ == "__main__":
    board_size = 5
    num_channels = 3

    # 随机生成一个棋盘 (1 个样本)
    board = np.random.randint(0, 2, size=(num_channels, board_size, board_size))
    policy = np.random.rand(board_size, board_size)
    value = np.array([0.5])
    weight = np.array([1.0])

    # 增强数据
    aug_boards, aug_policies, aug_values, aug_weights = augment_data(
        [board], [policy], value, weight
    )

    # 可视化
    visualize_augmented_boards(torch.from_numpy(board).float(), aug_boards)
