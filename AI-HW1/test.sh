#!/bin/bash

# Path to the directory containing the worlds
WORLD_DIR="./worlds"

# Test the first world with BFS
python3 main.py $WORLD_DIR 1 -e 0 -b 50 -a 2

# Test the second world with A* and Manhattan heuristic
python3 main.py $WORLD_DIR 4 -e 1 -b 50 -a 2

# Test the third world with Beam A* and Euclidean heuristic
python3 main.py $WORLD_DIR 3 -e 2 -b 100 -a 2

# Test with local search using Manhattan heuristic
python3 main.py $WORLD_DIR 1 -e 3 -b 50 -a 2

# Test with local search using Euclidean heuristic
python3 main.py $WORLD_DIR 1 -e 4 -b 50 -a 2
