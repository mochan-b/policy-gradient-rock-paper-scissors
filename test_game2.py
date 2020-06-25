
import numpy as np
import random

from game2 import play_tictactoe, two_player_improvement
from policy import softmax

def test_play_tictactoe():
    # Opponent always plays rock
    P = np.array([1, 0, 0])
    # Plays paper
    theta = np.array([float('-inf'), 1, float('-inf')], dtype=float)

    reward = play_tictactoe(P, theta, 100)
    assert reward == 100

    # Equally likely to choose an action of rock, paper or scissors
    theta = np.array([1, 1, 1], dtype=float)

    random.seed(0)
    reward = play_tictactoe(P, theta, 100)
    assert reward == -3

def test_two_player_improvement():
    # Plays paper
    #theta1 = np.array([float('-inf'), 1, float('-inf')], dtype=float)
    theta2 = np.array([0, 1, 1], dtype=float)
    # Just random with less chance for rock
    theta1 = np.array([1, 1, 0], dtype=float)

    t1, t2 = two_player_improvement(theta1, theta2, 100)