
import numpy as np

from model2 import State, Action, get_reward
from policy import softmax, softmax_diff

# Policy is a function that takes in a state and an action and gives the probability of that action
def policy (action, theta):
        action_probs = softmax(theta)
        return action_probs[action.value]

# Gradient of the policy at the given state and given action (using theta for the softmax policy)
def grad_policy (action, theta):
    grad = np.zeros(3)

    action_value = action.value

    for i in range(len(theta)):
        grad[i] = softmax_diff(theta, action_value, i)

    return grad

# Calculate the gradient of J for improving the gradient
def delta_J(theta, P):
    dJ = 0

    for a in Action:
        lv2 = 0
        for s in (State.ROCK, State.PAPER, State.SCISSORS):
            s_value = s.value
            lv2 += P[s_value] * get_reward(s, a)
        dJ += lv2 * grad_policy(a, theta)

    return dJ

# Improve the policy and return a new theta
def policy_improvement(alpha, n, theta, P):
    itheta = theta.copy()

    for i in range(n):
        dJ = delta_J(itheta, P)
        improvement = alpha * dJ
        itheta += improvement
    return itheta
