#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Rohan Singh
Alpha-beta minimax AI player for Othello.
"""

import numpy as np
from six.moves import input
from othello_shared import get_possible_moves, play_move, compute_utility


def max_value(state, player, alpha, beta):
    """ Returns the maximum utility for the maximizing player (dark = 1). """
    possible_moves = get_possible_moves(state, player)
    if not possible_moves:
        return compute_utility(state), None  # Terminal state

    max_eval, best_move = -float("inf"), None
    for move in possible_moves:
        new_state = play_move(state, player, move[0], move[1])
        eval_value, _ = min_value(new_state, 3 - player, alpha, beta)  # Switch player

        if eval_value > max_eval:
            max_eval, best_move = eval_value, move
        alpha = max(alpha, max_eval)
        if alpha >= beta:
            break  # Beta cutoff

    return max_eval, best_move


def min_value(state, player, alpha, beta):
    """ Returns the minimum utility for the minimizing player (light = 2). """
    possible_moves = get_possible_moves(state, player)
    if not possible_moves:
        return compute_utility(state), None  # Terminal state

    min_eval, best_move = float("inf"), None
    for move in possible_moves:
        new_state = play_move(state, player, move[0], move[1])
        eval_value, _ = max_value(new_state, 3 - player, alpha, beta)  # Switch player

        if eval_value < min_eval:
            min_eval, best_move = eval_value, move
        beta = min(beta, min_eval)
        if alpha >= beta:
            break  # Alpha cutoff

    return min_eval, best_move


def minimax(state, player):
    """ Calls max_value or min_value based on the player. """
    if player == 1:
        _, move = max_value(state, player, -float("inf"), float("inf"))
    else:
        _, move = min_value(state, player, -float("inf"), float("inf"))
    return move


def run_ai():
    """ Communication loop for running the AI with the game manager. """
    print("Minimax AI")
    color = int(input())

    while True:
        next_input = input()
        status, _, _ = next_input.strip().split()

        if status == "FINAL":
            print()
        else:
            board = np.array(eval(input()))
            movei, movej = minimax(board, color)
            print(f"{movei} {movej}")


if __name__ == "__main__":
    run_ai()
