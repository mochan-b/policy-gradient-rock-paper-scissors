import numpy as np
import random

from model2 import State, Action, get_reward
from policy2 import policy, grad_policy, delta_J, policy_improvement
from game2 import play_tictactoe

def test_get_reward():
    assert get_reward(State.READY, Action.ROCK) == 0
    assert get_reward(State.ROCK, Action.ROCK) == 0
    assert get_reward(State.PAPER, Action.ROCK) == -1
    assert get_reward(State.SCISSORS, Action.ROCK) == 1

def test_policy():
    # Equal probability for each of the actions
    theta = np.array([1, 1, 1], dtype=float)

    # Think rock. Probability of selecting rock
    p = policy(Action.ROCK, theta)
    assert p == 1/3

    # When predicting rock, always play paper
    theta = np.array([float("-inf"), 1, float("-inf")], dtype=float)
    p = policy(Action.ROCK, theta)
    assert p == 0
    p = policy(Action.PAPER, theta)
    assert p == 1

def test_grad_policy():
    theta = np.array([1, 1, 1])
    grad = grad_policy(Action.ROCK, theta)
    assert (grad == np.array([2/9, -1/9, -1/9])).all()
    grad = grad_policy(Action.PAPER, theta)
    assert (grad == np.array([-1/9, 2/9, -1/9])).all()
    grad = grad_policy(Action.SCISSORS, theta)
    assert (grad == np.array([-1/9, -1/9, 2/9])).all()

    theta = np.array([1, float("-inf"), float("-inf")])
    grad = grad_policy(Action.ROCK, theta)
    assert (grad == np.zeros((1, 3))).all()
    grad = grad_policy(Action.PAPER, theta)
    assert (grad == np.zeros((1, 3))).all()
    grad = grad_policy(Action.SCISSORS, theta)
    assert (grad == np.zeros((1, 3))).all()

def test_delta_J():
    # It is equally likely that the opponent draws rock, paper or scissors
    P = np.array([1 / 3, 1 / 3, 1 / 3])
    # Equally likely to choose an action of rock, paper or scissors
    theta = np.array([1, 1, 1])

    dJ = delta_J(theta, P)
    assert (dJ == np.zeros((1, 3))).all()

    # The opponent always plays rock
    P = np.array([1, 0, 0])
    # Equally likely to choose an action of rock, paper or scissors
    theta = np.array([1, 1, 1])

    dJ = delta_J(theta, P)
    assert (dJ == np.array([0, 1/3, -1/3])).all()

def test_policy_improvement():
    # The opponent always plays rock
    P = np.array([1, 0, 0])
    # Equally likely to choose an action of rock, paper or scissors
    theta = np.array([1, 1, 1], dtype=float)

    itheta = policy_improvement(1.0, 100, theta, P)

    # Play game with old policy
    random.seed(0)
    reward = play_tictactoe(P, theta, 100)
    ireward = play_tictactoe(P, itheta, 100)
    assert ireward > reward
