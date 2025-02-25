#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
This module contains functions that are accessed by the game manager
and by the each AI player.

Credit: Dr. Daniel Bauer
"""
import numpy as np

def find_lines(board, i, j, player):
    """
    Find all the uninterrupted lines of stones that would be captured if player
    plays column i and row j. 
    """
    lines = []
    for xdir, ydir in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], 
                       [-1, 0], [-1, 1]]:
        u = i
        v = j
        line = []

        u += xdir
        v += ydir
        found = False
        while 0 <= u < board.shape[0] and 0 <= v < board.shape[0]:
            if board[v, u] == 0:
                break
            elif board[v, u] == player:
                found = True
                break
            else: 
                line.append((u, v))
            u += xdir
            v += ydir
        if found and line: 
            lines.append(line)
    return lines
   

def get_possible_moves(board, player):
    """
    Return a list of all possible (column,row) tuples that player can play on
    the current board. 
    """
    result = []
    for i in range(board.shape[0]):
        for j in range(board.shape[0]):
            if board[j, i] == 0:
                lines = find_lines(board, i, j, player)
                if lines: 
                    result.append((i, j))
    return result


def play_move(board, player, i, j):
    new_board = np.copy(board)
    lines = find_lines(board, i, j, player)
    new_board[j, i] = player
    for line in lines: 
        for u, v in line:
            new_board[v, u] = player
    return new_board


def get_score(board):
    p1_count = 0
    p2_count = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[0]):
            if board[i, j] == 1:
                p1_count += 1
            elif board[i, j] == 2:
                p2_count += 1
    return p1_count, p2_count


def compute_utility(board):
    p1, p2 = get_score(board)
    return p1 - p2
