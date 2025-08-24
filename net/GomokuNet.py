import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 搭建残差块
# 参考 https://github.com/tensorfly-gpu/aichess.git
class ResBlock(nn.Module):

    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
        )
        self.conv1_bn = nn.BatchNorm2d(
            num_filters,
        )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
        )
        self.conv2_bn = nn.BatchNorm2d(
            num_filters,
        )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)


class PolicyValueNet(nn.Module):
    def __init__(self, in_channels=5, board_size=6):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.block = nn.Sequential(
            ResBlock(num_filters=256),
            ResBlock(num_filters=256),
            ResBlock(num_filters=256),
            ResBlock(num_filters=256),
            ResBlock(num_filters=256),
            ResBlock(num_filters=256),
            ResBlock(num_filters=256),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=16, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=16 * board_size * board_size,
                out_features=board_size * board_size,
            ),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=8, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=8 * board_size * board_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # 共享特征提取
        x = self.conv1(x)
        x = self.block(x)
        # --- Policy head ---
        p = self.policy_head(x)
        p = p.reshape(p.shape[0], self.board_size, self.board_size)

        # --- Value head ---
        v = self.value_head(x)

        return p, v

    def calc_one_board(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        device = next(self.parameters()).device
        x = x.unsqueeze(0).to(device, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            logits, value = self.forward(x)
            probs = (
                F.softmax(logits, dim=1)
                .view(-1, self.board_size, self.board_size)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            return probs, value.squeeze(0).detach().cpu().numpy()


if __name__ == "__main__":
    model = PolicyValueNet()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    data = torch.randn(16, 5, 6, 6)
    probs, value = model(data)
    print(probs.shape)
    print(value.shape)
