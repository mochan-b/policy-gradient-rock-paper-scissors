
import numpy as np

from enum import IntEnum

# These are the states that we are present in the game
class State(IntEnum):
    READY = -1,     # Think the opponent will play rock
    ROCK = 0,      # Opponent played rock
    PAPER = 1,     # Opponent played paper
    SCISSORS = 2,  # Opponent player scissors

# These are the actions that are available in the game
class Action(IntEnum):
    ROCK = 0,
    PAPER = 1,
    SCISSORS = 2,

# Reward matrices.
# Rows are for what we played. Columns are for what the opponent played
R = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])

# With the given reward matrix, get the reward for the given action and state
def get_reward(s, a):
    if s == State.READY:
        return 0
    else:
        return R[a.value][s.value]
