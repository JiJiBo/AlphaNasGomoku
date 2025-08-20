import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class PolicyValueNet(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, num_blocks=5, value_dim=128, board_size=19):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        self.policy_conv1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.policy_bn1 = nn.BatchNorm2d(hidden_channels // 2)
        self.policy_conv2 = nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, padding=1)

        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, value_dim)
        self.value_fc2 = nn.Linear(value_dim, 1)

    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))

        for block in self.res_blocks:
            x = block(x)

        policy = F.relu(self.policy_bn1(self.policy_conv1(x)))
        policy = self.policy_conv2(policy)
        policy = policy.squeeze(1)
        policy_logits = policy.view(x.size(0), -1)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(x.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

    def calc_one_board(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        device = next(self.parameters()).device
        x = x.unsqueeze(0).to(device, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            logits, value  = self.forward(x)
            probs = F.softmax(logits, dim=1).view(-1, self.board_size, self.board_size).squeeze(0).detach().cpu().numpy()
            return probs, value.squeeze(0).detach().cpu().numpy()


if __name__ == '__main__':
    model = PolicyValueNet()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
