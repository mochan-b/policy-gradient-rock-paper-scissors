
import numpy as np
import random
import bisect

from model import State, Action, get_reward
from policy import policy

def sample_from_dist(d):
    cdf = np.cumsum(d)
    p = random.random()
    return bisect.bisect(cdf, p)

# Play the game given the transition probability P, theta for the policy parameters
# and the number of games.
# Returns the total reward
def play_tictactoe(P, theta, d, n):
    total_reward = 0
    for i in range(n):
        # Choose the starting state
        s = sample_from_dist(d)
        state = State(s)

        # Get the action
        action_probs = np.zeros((3,))
        for action in Action:
            action_probs[action.value] = policy(state, action, theta)
        a = sample_from_dist(action_probs)
        action = Action(a)

        # Get the transition state
        s2 = sample_from_dist(P[s])
        state2 = State(s2 + 3)

        # Get the reward
        reward = get_reward(state2, action)
        total_reward += reward

    return total_reward