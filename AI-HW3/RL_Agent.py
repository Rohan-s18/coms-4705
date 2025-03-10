"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A Q-learning agent for a stochastic task environment

author: rohan singh (rs4607@columbia.edu)
"""

import random
import math
import sys


class RL_Agent(object):

    def __init__(self, states, valid_actions, parameters):
        self.alpha = parameters["alpha"]
        self.epsilon = parameters["epsilon"]
        self.gamma = parameters["gamma"]
        self.Q0 = parameters["Q0"]

        self.states = states
        self.Qvalues = {}
        for state in states:
            for action in valid_actions(state):
                self.Qvalues[(state, action)] = parameters["Q0"]


    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        self.alpha = alpha


    def choose_action(self, state, valid_actions):
        """ Choose an action using epsilon-greedy selection.

        Args:
            state (tuple): Current robot state.
            valid_actions (list): A list of possible actions.
        Returns:
            action (string): Action chosen from valid_actions.
        """
        # TODO
        return None


    def update(self, state, action, reward, successor, valid_actions):
        """ Update self.Qvalues for (state, action) given reward and successor.

        Args:
            state (tuple): Current robot state.
            action (string): Action taken at state.
            reward (float): Reward given for transition.
            successor (tuple): Successor state.
            valid_actions (list): A list of possible actions at successor state.
        """
        # TODO
        pass
