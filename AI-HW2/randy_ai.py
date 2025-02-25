#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Randy is an "AI" for Othello that randomly chooses a legal move. Play against
this AI to get familiar with the game. You can also have your AI compete
against your AI to test its performance. 

Credit: Dr. Daniel Bauer
"""

import random
import time
import numpy as np
from six.moves import input
from othello_shared import get_possible_moves


def select_move(board, color):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  
    """
    moves = get_possible_moves(board, color)
    if not moves:
        return None
    print(moves)
    i, j = random.choice(moves)
    time.sleep(0.1)
    return i, j




def run_ai():
    """
    This function establishes communication with the game manager. 
    It first introduces itself and receives its color. 
    Then it repeatedly receives the current score and current board state
    until the game is over. 
    """
    print("Randy")          # First line is the name of this AI
    color = int(input())    # 1 for dark (first), 2 for light (second)

    while True:
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL":
            print 
        else:
            board = np.array(eval(input()))
            movei, movej = select_move(board, color)
            print("{} {}".format(movei, movej)) 


if __name__ == "__main__":
    run_ai()
