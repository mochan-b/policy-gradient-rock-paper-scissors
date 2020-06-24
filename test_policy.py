
import numpy as np
import random

from policy import softmax, policy, softmax_diff_fragment, softmax_diff, grad_policy, delta_J, policy_improvement
from model import State, Action, get_reward
from game import play_tictactoe

def test_get_reward():
    assert get_reward(State.ROCK_1, Action.ROCK) == 0
    assert get_reward(State.ROCK_2, Action.ROCK) == 0
    assert get_reward(State.PAPER_2, Action.ROCK) == -1
    assert get_reward(State.SCISSORS_2, Action.ROCK) == 1

def test_softmax():
    x = [1, 1]
    p = softmax(x)
    assert (p == [0.5] * 2).all()

    x = [1, 1, 1, 1]
    p = softmax(x)
    assert (p == [0.25] * 4).all()

def test_policy():
    # Equal probability for each of the actions
    theta = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)

    # Think rock. Probability of selecting rock
    p = policy(State.ROCK_1, Action.ROCK, theta)
    assert p == 1/3

    # When predicting rock, always play paper
    theta = np.array([[float("-inf"), 1, float("-inf")], [1, 1, 1], [1, 1, 1]], dtype=float)
    p = policy(State.ROCK_1, Action.ROCK, theta)
    assert p == 0
    p = policy(State.ROCK_1, Action.PAPER, theta)
    assert p == 1

def test_softmax_diff_fragment():
    x = [1, 1, 1]
    p = softmax_diff_fragment(x, 1, 2)
    assert p == 1/9

def test_softmax_diff():
    x = [1, 1, 1]
    p = softmax_diff(x, 1, 1)
    assert p == 2/9
    p = softmax_diff(x, 1, 2)
    assert p == -1/9

def test_grad_policy():
    theta = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    grad = grad_policy(State.ROCK_1, Action.ROCK, theta)
    assert (grad == np.array([2/9, -1/9, -1/9, 0, 0, 0, 0, 0, 0])).all()
    grad = grad_policy(State.ROCK_1, Action.PAPER, theta)
    assert (grad == np.array([-1/9, 2/9, -1/9, 0, 0, 0, 0, 0, 0])).all()
    grad = grad_policy(State.ROCK_1, Action.SCISSORS, theta)
    assert (grad == np.array([-1/9, -1/9, 2/9, 0, 0, 0, 0, 0, 0])).all()

    theta = np.array([[1, float("-inf"), float("-inf")], [1, 1, 1], [1, 1, 1]])
    grad = grad_policy(State.ROCK_1, Action.ROCK, theta)
    assert (grad == np.zeros((1, 9))).all()
    grad = grad_policy(State.ROCK_1, Action.PAPER, theta)
    assert (grad == np.zeros((1, 9))).all()
    grad = grad_policy(State.ROCK_1, Action.SCISSORS, theta)
    assert (grad == np.zeros((1, 9))).all()

def test_delta_J():
    # It is equally likely that the opponent draws rock, paper or scissors
    P = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
    # Equally think the opponent will draw rock, paper or scissors
    d = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0, 0])
    # Equally likely to choose an action of rock, paper or scissors
    theta = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    dJ = delta_J(theta, P, d)
    assert (dJ == np.zeros((1, 9))).all()

    # The opponent always plays rock
    P = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    # Equally think the opponent will draw rock, paper or scissors
    d = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0, 0])
    # Equally likely to choose an action of rock, paper or scissors
    theta = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    dJ = delta_J(theta, P, d)
    assert (dJ == np.array([0, 1/9, -1/9, 0, 1/9, -1/9, 0, 1/9, -1/9])).all()

    # The opponent always plays what we think they will play
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Equally think the opponent will draw rock, paper or scissors
    d = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0, 0])
    # Equally likely to choose an action of rock, paper or scissors
    theta = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    dJ = delta_J(theta, P, d)
    assert (dJ == np.array([0, 1 / 9, -1 / 9, -1/9, 0, 1 / 9, 1/9, -1/9, 0])).all()

def test_policy_improvement():
    # The opponent always plays what we think they will play
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Equally think the opponent will draw rock, paper or scissors
    d = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0, 0])
    # Equally likely to choose an action of rock, paper or scissors
    theta = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)

    itheta = policy_improvement(1.0, 100, theta, P, d)

    # Play game with old policy
    random.seed(0)
    reward = play_tictactoe(P, theta, d, 100)
    ireward = play_tictactoe(P, itheta, d, 100)
    assert ireward > reward
