from net.GomokuNet import PolicyValueNet


class MCTSAgent:
    def __init__(self, model: PolicyValueNet, use_rand=0.1, c_puct=1.4):
        self.model = model
        self.use_rand = use_rand
        self.c_puct = c_puct
