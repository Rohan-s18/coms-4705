#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
Author: Rohan Singh
Monte Carlo Tree Search (MCTS) AI for Othello.
"""

import numpy as np
import random
from othello_shared import get_possible_moves, play_move, compute_utility


class Node:
    def __init__(self, state, player, parent=None, children=None, value=0, visits=1):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = children if children is not None else []
        self.value = value  # Initial value
        self.visits = visits  # Avoid division by zero in UCT

    def is_fully_expanded(self):
        return len(self.children) == len(get_possible_moves(self.state, self.player))

    def best_child(self, exploration_weight=1.41):
        return max(self.children, key=lambda node: (node.value / node.visits) + 
                   exploration_weight * np.sqrt(np.log(self.visits) / node.visits))


def select(node, player):
    """ Select the best node based on UCT until an expandable node is found. """
    while node.is_fully_expanded() and node.children:
        node = node.best_child()
    return node


def expand(node):
    """ Expand a child node from available moves. """
    possible_moves = get_possible_moves(node.state, node.player)
    existing_moves = {tuple(child.state.flatten()) for child in node.children}

    for move in possible_moves:
        new_state = play_move(node.state, node.player, *move)
        if tuple(new_state.flatten()) not in existing_moves:  # Avoid duplicate expansions
            child = Node(new_state, 3 - node.player, parent=node)
            node.children.append(child)
            return child

    return random.choice(node.children)  # In case all moves are already expanded


def simulate(node):
    """ Simulate a random rollout from the given node. """
    state = np.copy(node.state)
    player = node.player

    for _ in range(10):  # Limit depth to avoid infinite loops
        possible_moves = get_possible_moves(state, player)
        if not possible_moves:
            break
        move = random.choice(possible_moves)
        state = play_move(state, player, *move)
        player = 3 - player  # Swap player

    return compute_utility(state)


def backprop(node, result):
    """Backpropagate the simulation result up the tree."""
    while node is not None:

        node.visits += 1  # Increment visit count

        # Correct value adjustment based on the player
        if node.player == 1:  # Player 1 (dark)
            node.value -= result  # Add result for Player 1
        else:  # Player 2 (light)
            node.value += result  # Subtract result for Player 2

        node = node.parent  # Move up the tree


def mcts_2(initial_state, player, itermax=1000):
    """ Perform MCTS to find the best move. """
    root = Node(initial_state, player)

    for _ in range(itermax):
        node = select(root, player)
        if not node.is_fully_expanded():
            node = expand(node)
        result = simulate(node)
        backprop(node, result)

    return max(root.children, key=lambda n: n.visits).state  # Best move is most visited


def mcts(initial_state, player, itermax=1000):
    """ Perform MCTS to find the best move. """
    root = Node(initial_state, player)

    for _ in range(itermax):
        node = select(root, player)
        if not node.is_fully_expanded():
            node = expand(node)
        result = simulate(node)
        backprop(node, result)

    if not root.children:
        return None  # No valid move available, return None

    return max(root.children, key=lambda n: n.visits).state  # Best move is most visited




####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("MCTS AI")        # First line is the name of this AI
    color = int(input())    # 1 for dark (first), 2 for light (second)

    while True:
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL":
            print()
        else:
            board = np.array(eval(input()))
            movei, movej = mcts(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
