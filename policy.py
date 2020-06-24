
import numpy as np

from model import State, Action, get_reward

# Softmax policy
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Policy is a function that takes in a state and an action and gives the probability of that action
def policy (state, action, theta):
    if state == State.ROCK_1 or state == State.PAPER_1 or state == State.SCISSORS_1:
        theta_for_state = theta[state.value]
        action_probs = softmax(theta_for_state)
        return action_probs[action.value]
    else:
        return 0

# Represents a term in the derivative. We compose them together to form the derivative later on
# Here i and j are the top exponents and the sum is over all of the items below
def softmax_diff_fragment(x, i, j):
    return np.exp(x[i] + x[j]) / np.square( np.sum(np.exp(x), axis=0) )

# Get the derivative of a softmax policy
# i is the term that is on the top of the softmax. di is the one we are taking the derivative of
def softmax_diff (x, i, di):
    if i==di:
        diff = 0
        for j in range(len(x)):
            if j != i:
                diff += softmax_diff_fragment(x, i, j)
        return diff
    else:
        return -softmax_diff_fragment(x, di, i)

# Gradient of the policy at the given state and given action (using theta for the softmax policy)
def grad_policy (state, action, theta):
    grad = np.zeros(9)

    if state == State.ROCK_1 or state == State.PAPER_1 or state == State.SCISSORS_1:
        state_value = state.value
        action_value = action.value

        theta_state = theta[state.value].copy()

        grad_frag = np.zeros(theta_state.shape)
        for i in range(len(theta_state)):
            grad_frag[i] = softmax_diff(theta_state, action_value, i)

        grad[state_value*3:((state_value+1)*3)] = grad_frag
        return grad
    else:
        return grad

# Calculate the gradient of J for improving the gradient
def delta_J(theta, P, d):
    dJ = 0

    for s in (State.ROCK_1, State.PAPER_1, State.SCISSORS_1):
        s_value = s.value
        lv1 = 0
        for a in Action:
            lv2 = 0
            for sp in (State.ROCK_2, State.PAPER_2, State.SCISSORS_2):
                sp_value = sp.value - 3
                lv2 += P[s_value, sp_value] * get_reward(sp, a)
            lv1 += lv2 * grad_policy(s, a, theta)
        dJ += d[s_value] * lv1

    return dJ

# Improve the policy and return a new theta
def policy_improvement(alpha, n, theta, P, d):
    itheta = theta.copy()

    for i in range(n):
        dJ = delta_J(itheta, P, d)
        improvement = alpha * (dJ.reshape(itheta.shape))
        itheta += improvement
    return itheta
