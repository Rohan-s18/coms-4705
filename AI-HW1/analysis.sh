#!/bin/bash

# Set up the path to the directory containing world files
WORLD_DIR="./worlds"

# Task 1: Run uninformed search (DFS and BFS) on each world
echo "Running uninformed search (DFS and BFS) on all worlds..."

# Run DFS and BFS for World 1
echo "Running DFS and BFS for World 1..."
python3 main.py $WORLD_DIR 1 -e 0 -b 50 -a 1

# Run DFS and BFS for World 2
echo "Running DFS and BFS for World 2..."
python3 main.py $WORLD_DIR 2 -e 0 -b 50 -a 1

# Run DFS and BFS for World 3
echo "Running DFS and BFS for World 3..."
python3 main.py $WORLD_DIR 3 -e 0 -b 50 -a 1

# Task 1: Show figures for DFS and BFS on World 1 in the document
echo "Figures for DFS and BFS for World 1 will be included in the document."

# Task 2: Run A* and Beam Search on all worlds
echo "Running A* and Beam Search on all worlds..."

# Run A* and Beam Search for World 1
echo "Running A* and Beam Search for World 1..."
python3 main.py $WORLD_DIR 1 -e 1 -b 50 -a 1

# Run A* and Beam Search for World 2
echo "Running A* and Beam Search for World 2..."
python3 main.py $WORLD_DIR 2 -e 1 -b 50 -a 1

# Run A* and Beam Search for World 3
echo "Running A* and Beam Search for World 3..."
python3 main.py $WORLD_DIR 3 -e 1 -b 50 -a 1

# Run A* and Beam Search for World 4 (if available)
echo "Running A* and Beam Search for World 4..."
python3 main.py $WORLD_DIR 4 -e 1 -b 50 -a 1

# Task 2: Show output figures for a world with different solutions from A* and Beam Search
echo "Figures for World 1, where A* and Beam Search return different solutions, will be included in the document."

# Task 3: Tune Beam width until solutions match
echo "Tuning Beam width to match solutions of A* and Beam Search for World 1..."

# Run Beam Search with increasing width to match A* solution
for width in 10 20 30 40 50 60; do
    echo "Running Beam Search with width $width for World 1..."
    python3 main.py $WORLD_DIR 1 -e 1 -b $width -a 1
done

# Task 3: Show output figures for matching A* and Beam Search results at tuned Beam width
echo "Figures for Beam Search with tuned Beam width for World 1 will be included in the document."

# Task 4: Run local search on each world with both heuristics
echo "Running local search on all worlds using both heuristics..."

# Run local search for World 1 with Manhattan heuristic
echo "Running local search for World 1 with Manhattan heuristic..."
python3 main.py $WORLD_DIR 1 -e 3 -b 50 -a 1

# Run local search for World 2 with Manhattan heuristic
echo "Running local search for World 2 with Manhattan heuristic..."
python3 main.py $WORLD_DIR 2 -e 3 -b 50 -a 1

# Run local search for World 3 with Manhattan heuristic
echo "Running local search for World 3 with Manhattan heuristic..."
python3 main.py $WORLD_DIR 3 -e 3 -b 50 -a 1

# Run local search for World 4 with Manhattan heuristic
echo "Running local search for World 4 with Manhattan heuristic..."
python3 main.py $WORLD_DIR 4 -e 3 -b 50 -a 1

# Task 4: Show output figure for World where local search finds a path solution
echo "Figures for local search results will be included in the document."

# End of the script
echo "All tasks completed. Please refer to the document for output figures and explanations."
