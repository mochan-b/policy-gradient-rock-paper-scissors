
import numpy as np

from enum import IntEnum

# These are the states that we are present in the game
class State(IntEnum):
    ROCK_1 = 0,     # Think the opponent will play rock
    PAPER_1 = 1,    # Think the opponent will play paper
    SCISSORS_1 = 2, # Think the opponent will play scissors
    ROCK_2 = 3,     # Opponent played rock
    PAPER_2 = 4,    # Opponent played paper
    SCISSORS_2 = 5, # Opponent player scissors

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
    if s == State.ROCK_2 or s == State.PAPER_2 or s == State.SCISSORS_2:
        return R[a.value][s.value - 3]
    else:
        return 0
