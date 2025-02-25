import numpy as np
import random
from collections import namedtuple

# Import the functions from each AI script
from mcts_ai import mcts  # Assuming mcts is the function to get the best move using MCTS
from minimax_ai import minimax  # Assuming minimax is the function to get the best move using Minimax
from randy_ai import select_move  # Assuming select_move is the function to get the random move from Randy

# Define the state and board size
Board = namedtuple('Board', ['state', 'size'])

# Placeholder for game utility functions
def get_possible_moves(board, color):
    # Returns a list of valid moves (in the form of (x, y) positions)
    # Placeholder implementation
    return [(random.randint(0, 3), random.randint(0, 3)) for _ in range(5)]

def run_game(player1_fn, player2_fn, board_size=4):
    """
    Run a game between two players on a given board size.
    player1_fn and player2_fn are functions representing the two agents' moves.
    Returns the winner: 1 for player1, 2 for player2, 0 for draw.
    """
    # Initialize an empty board
    board = np.zeros((board_size, board_size), dtype=int)
    turn = 1  # Player 1 starts
    game_over = False

    while not game_over:

        # Check for end condition (simple version, e.g., full board or specific conditions)
        if np.all(board != 0):  # Check for full board
            game_over = True
        
        # Get the current player move
        if turn == 1:
            move = player1_fn(board, turn)
        else:
            move = player2_fn(board, turn)

        # Placeholder for making a move on the board
        board[move] = turn  # Update board with current player's move

        

        # Switch turn
        turn = 3 - turn  # 1 -> 2, 2 -> 1

    # Determine the winner (just a placeholder logic for now)
    player1_score = np.sum(board == 1)
    player2_score = np.sum(board == 2)
    
    if player1_score > player2_score:
        return 1  # Player 1 wins
    elif player2_score > player1_score:
        return 2  # Player 2 wins
    else:
        return 0  # Draw


# 1. Test two minimax agents against each other on a 4x4 board
def test_minimax_vs_minimax(board_size=4):
    print("Running minimax vs minimax on a 4x4 board")
    winner = run_game(minimax, minimax, board_size)
    if winner == 0:
        print("Draw")
    elif winner == 1:
        print("Minimax (Player 1) wins")
    else:
        print("Minimax (Player 2) wins")

# 2. Test minimax agent against Randy on a 4x4 board (Player 1 and Player 2)
def test_minimax_vs_randy(board_size=4):
    print("Running minimax vs randy on a 4x4 board with player 1 starting")
    winner = run_game(minimax, select_move, board_size)
    if winner == 0:
        print("Draw")
    elif winner == 1:
        print("Minimax (Player 1) wins")
    else:
        print("Randy (Player 2) wins")

    print("Running minimax vs randy on a 4x4 board with player 2 starting")
    winner = run_game(select_move, minimax, board_size)
    if winner == 0:
        print("Draw")
    elif winner == 1:
        print("Randy (Player 1) wins")
    else:
        print("Minimax (Player 2) wins")

# 3. Test minimax agent against MCTS on a 4x4 board
def test_minimax_vs_mcts(board_size=4):
    print("Running minimax vs MCTS on a 4x4 board with player 1 starting")
    winner = run_game(minimax, mcts, board_size)
    if winner == 0:
        print("Draw")
    elif winner == 1:
        print("Minimax (Player 1) wins")
    else:
        print("MCTS (Player 2) wins")

    print("Running minimax vs MCTS on a 4x4 board with player 2 starting")
    winner = run_game(mcts, minimax, board_size)
    if winner == 0:
        print("Draw")
    elif winner == 1:
        print("MCTS (Player 1) wins")
    else:
        print("Minimax (Player 2) wins")

# 4. Test MCTS agent against Randy on a 6x6 board (Player 1 and Player 2)
def test_mcts_vs_randy(board_size=6):
    print("Running MCTS vs Randy on a 6x6 board with player 1 starting")
    winner = run_game(mcts, select_move, board_size)
    if winner == 0:
        print("Draw")
    elif winner == 1:
        print("MCTS (Player 1) wins")
    else:
        print("Randy (Player 2) wins")

    print("Running MCTS vs Randy on a 6x6 board with player 2 starting")
    winner = run_game(select_move, mcts, board_size)
    if winner == 0:
        print("Draw")
    elif winner == 1:
        print("Randy (Player 1) wins")
    else:
        print("MCTS (Player 2) wins")


# Run the analysis
def run_analysis():
    # Test minimax vs minimax
    test_minimax_vs_minimax()

    # Test minimax vs Randy (Player 1 and Player 2)
    test_minimax_vs_randy()

    # Test minimax vs MCTS (Player 1 and Player 2)
    test_minimax_vs_mcts()

    # Test MCTS vs Randy on a 6x6 board (Player 1 and Player 2)
    test_mcts_vs_randy()

if __name__ == "__main__":
    run_analysis()

