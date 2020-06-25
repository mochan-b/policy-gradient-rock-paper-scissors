
import numpy as np
import random
import bisect

from model2 import State, Action, get_reward
from policy2 import softmax, policy_improvement
from game import sample_from_dist

# Play the game given the transition probability P, theta for the policy parameters
# and the number of games.
# Returns the total reward
def play_tictactoe(P, theta, n):
    total_reward = 0
    for i in range(n):
        # Get the action
        action_probs = softmax(theta)
        a = sample_from_dist(action_probs)
        action = Action(a)

        # Get the opponent's move
        s = sample_from_dist(P)
        state = State(s)

        # Get the reward
        reward = get_reward(state, action)
        total_reward += reward

    return total_reward

# Two player improvement game
def two_player_improvement(theta1, theta2, n):
    t1 = theta1.copy()
    t2 = theta2.copy()

    for i in range(n):
        P2 = softmax(t2)
        t1 = policy_improvement(1.0, 10, t1, P2)
        P1 = softmax(t1)
        t2 = policy_improvement(1.0, 10, t2, P1)

        # Yo-yos between policies of choosing one mostly to choosing another one
        print(P2, P1)

    return t1, t2