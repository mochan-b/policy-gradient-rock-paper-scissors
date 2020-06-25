
import numpy as np
import random

from game import sample_from_dist, play_tictactoe
from model import State, Action

def test_state():
    assert State(0) == State.ROCK_1

def test_cdf():
    pdf = np.array([0.8, 0.1, 0.1])
    cdf = np.cumsum(pdf)
    assert (cdf == np.array([0.8, 0.9, 1.0])).all()

def test_sample_from_d():
    d = np.array([0.8, 0.1, 0.1])

    selections = [0, 0, 0]
    random.seed(1)
    for i in range(100):
        j = sample_from_dist(d)
        selections[j] += 1

    assert (selections == [80, 11, 9])

def test_play_tictactoe():
    # Opponent always plays what we think they will play
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Equally think the opponent will draw rock, paper or scissors
    d = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0, 0])
    # Equally likely to choose an action of rock, paper or scissors
    theta = np.array([[float('-inf'), 1, float('-inf')], [float('-inf'), float('-inf'), 1], [1, float('-inf'), float('-inf')]], dtype=float)

    reward = play_tictactoe(P, theta, d, 100)
    assert reward == 100

    # The opponent always plays what we think they will play
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Equally think the opponent will draw rock, paper or scissors
    d = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0, 0])
    # Equally likely to choose an action of rock, paper or scissors
    theta = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)

    random.seed(0)
    reward = play_tictactoe(P, theta, d, 100)
    assert reward == 6