from torch.utils.data import Dataset


class Weighted_Dataset(Dataset):
    def __init__(self, boards, policies, values, weights):
        self.boards = boards
        self.policies = policies
        self.values = values
        self.weights = weights

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.policies[idx], self.values[idx], self.weights[idx]
