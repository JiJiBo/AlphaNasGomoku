import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyValueNet(nn.Module):
    def __init__(self, in_channels=3, board_size=6):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size

        # 公共卷积层 (3层)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # --- Policy head ---
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)  # 降维
        self.policy_fc = nn.Linear(4 * board_size * board_size,
                                   board_size * board_size)

        # --- Value head ---
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)  # 降维
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 共享特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # --- Policy head ---
        p = F.relu(self.policy_conv(x))
        p = p.view(-1, 4 * self.board_size * self.board_size)
        policy_logits = self.policy_fc(p)

        # --- Value head ---
        v = F.relu(self.value_conv(x))
        v = v.view(-1, 2 * self.board_size * self.board_size)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def calc_one_board(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        device = next(self.parameters()).device
        x = x.unsqueeze(0).to(device, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            logits, value = self.forward(x)
            probs = F.softmax(logits, dim=1).view(
                -1, self.board_size, self.board_size
            ).squeeze(0).detach().cpu().numpy()
            return probs, value.squeeze(0).detach().cpu().numpy()


if __name__ == '__main__':
    model = PolicyValueNet()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
