"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A dynamic programming agent for a stochastic task environment

author: rohan singh (rs4607@columbia.edu)
"""

import random
import math
import sys


class DP_Agent(object):

    def __init__(self, states, parameters):
        self.gamma = parameters["gamma"]
        self.V0 = parameters["V0"]

        self.states = states
        self.values = {}
        self.policy = {}

        for state in states:
            self.values[state] = parameters["V0"]
            self.policy[state] = None


    def setEpsilon(self, epsilon):
        pass

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        pass


    def choose_action(self, state, valid_actions):
        return self.policy[state]

    def update(self, state, action, reward, successor, valid_actions):
        pass


    def policy_evaluation(self, transition):
        """ Computes all values for current policy by iteration and stores them in self.values.
        Args:
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        # TODO
        pass


    def policy_extraction(self, valid_actions, transition):
        """ Computes all optimal actions using value iteration and stores them in self.policy.
        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        # TODO
        pass


    def policy_iteration(self, valid_actions, transition):
        """ Runs policy iteration to learn an optimal policy. Calls policy_evaluation() and policy_extraction().
        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        # TODO
        pass