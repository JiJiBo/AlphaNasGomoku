# 落子者
from enum import Enum, unique

PLAYER_BLACK = 1
PLAYER_WHITE = -1
PLAYER_EMPTY = 0

KL_TARG = 0.02


# 赢棋者
@unique
class Winner(Enum):
    EMPTY = PLAYER_EMPTY
    BLACK = PLAYER_BLACK
    WHITE = PLAYER_WHITE


if __name__ == "__main__":
    print(Winner.WHITE.value.real)
