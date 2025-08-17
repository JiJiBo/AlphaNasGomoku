# 落子者
from enum import Enum, unique

PLAYER_BLACK = 1
PLAYER_WHITE = 2
PLAYER_EMPTY = 0
# 赢棋者
@unique
class Winner(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
