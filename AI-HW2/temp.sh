#!/bin/bash

# Function to run a game between two AIs
run_game() {
  agent1=$1
  agent2=$2
  board_size=$3
  player1_start=$4
  echo "Running $agent1 vs $agent2 on a $board_size board with player 1 starting: $player1_start"

  # Initialize a 4x4 or 6x6 board
  if [[ $board_size -eq 4 ]]; then
    board="[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]"
  elif [[ $board_size -eq 6 ]]; then
    board="[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]"
  fi

  # Run the first agent
  if [[ $player1_start -eq 1 ]]; then
    # Player 1 starts with agent1
    echo "Running $agent1 (Player 1) vs $agent2 (Player 2)"
    python3 "$agent1.py" <<EOF
$player1_start
$board
EOF
    # Run the second agent
    python3 "$agent2.py" <<EOF
$((3 - player1_start))
$board
EOF
  fi
}

# Experiment 1: Test two Minimax agents against each other on a 4x4 board
minimax_vs_minimax() {
  run_game "minimax_ai" "minimax_ai" 4 1
}

# Experiment 2: Test Minimax vs Randy on a 4x4 board
minimax_vs_randy() {
  run_game "minimax_ai" "randy_ai" 4 1
}

# Experiment 3: Test Minimax vs MCTS on a 4x4 board
minimax_vs_mcts() {
  run_game "minimax_ai" "mcts_ai" 4 1
}

# Experiment 4: Test MCTS vs Randy on a 6x6 board
mcts_vs_randy() {
  run_game "mcts_ai" "randy_ai" 6 1
}

# Running the experiments
minimax_vs_minimax
minimax_vs_randy
minimax_vs_mcts
mcts_vs_randy
