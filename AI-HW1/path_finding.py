import numpy as np
from queue import PriorityQueue
from collections import deque
from utils.utils import PathPlanMode, Heuristic, cost, expand, visualize_expanded, visualize_path


def compute_heuristic(node, goal, heuristic: Heuristic):
    """ Computes an admissible heuristic value of node relative to goal. """
    if heuristic == Heuristic.MANHATTAN:
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    elif heuristic == Heuristic.EUCLIDEAN:
        return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)
    return 0


def get_neighbors(node, grid):
    """ Returns valid neighbors of a node in the grid. Assumes 4-connected grid. """
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for dx, dy in directions:
        neighbor = (node[0] + dx, node[1] + dy)
        if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor[0], neighbor[1]] != 1:
            neighbors.append(neighbor)
    return neighbors


def uninformed_search(grid, start, goal, mode: PathPlanMode):
    """ Find a path from start to goal using BFS or DFS. """
    frontier = deque([start]) if mode == PathPlanMode.BFS else [start]
    frontier_sizes = []
    expanded = []
    reached = {start: None}

    while frontier:
        if mode == PathPlanMode.BFS:
            node = frontier.popleft()
        else:  # DFS
            node = frontier.pop()
        
        if node == goal:
            # Reconstruct path
            path = []
            while node is not None:
                path.append(node)
                node = reached[node]
            path.reverse()
            return path, expanded, frontier_sizes

        expanded.append(node)
        neighbors = get_neighbors(node, grid)
        for neighbor in neighbors:
            if neighbor not in reached:
                frontier.append(neighbor)
                reached[neighbor] = node

        frontier_sizes.append(len(frontier))

    return [], expanded, frontier_sizes


def a_star(grid, start, goal, mode: PathPlanMode, heuristic: Heuristic, width):
    """ Perform A* or Beam search to find the shortest path. """
    frontier = PriorityQueue()
    frontier.put((0 + compute_heuristic(start, goal, heuristic), 0, start))  # (f, g, node)
    frontier_sizes = []
    expanded = []
    reached = {start: {"cost": cost(grid, start), "parent": None}}

    while not frontier.empty():
        _, g, node = frontier.get()

        if node == goal:
            # Reconstruct path
            path = []
            while node is not None:
                path.append(node)
                node = reached[node]["parent"]
            path.reverse()
            return path, expanded, frontier_sizes

        expanded.append(node)
        neighbors = get_neighbors(node, grid)
        for neighbor in neighbors:
            new_cost = g + cost(grid, neighbor)
            if neighbor not in reached or new_cost < reached[neighbor]["cost"]:
                reached[neighbor] = {"cost": new_cost, "parent": node}
                f = new_cost + compute_heuristic(neighbor, goal, heuristic)
                frontier.put((f, new_cost, neighbor))

        frontier_sizes.append(len(frontier.queue))

    return [], expanded, frontier_sizes


def local_search(grid, start, goal, heuristic: Heuristic):
    """ Find a path from start to goal using local search (greedy). """
    path = [start]
    current_node = start

    while current_node != goal:
        neighbors = get_neighbors(current_node, grid)
        best_node = None
        best_heuristic = float('inf')

        for neighbor in neighbors:
            h = compute_heuristic(neighbor, goal, heuristic)
            if h < best_heuristic:
                best_heuristic = h
                best_node = neighbor

        if best_node:
            path.append(best_node)
            current_node = best_node
        else:
            break  # No valid neighbors to proceed

    return path


def test_world(world_id, start, goal, h, width, animate, world_dir):
    print(f"Testing world {world_id}")
    grid = np.load(f"{world_dir}/world_{world_id}.npy")

    if h == 0:
        modes = [
            PathPlanMode.DFS,
            PathPlanMode.BFS
        ]
        print("Modes: 1. DFS, 2. BFS")
    elif h == 1 or h == 2:
        modes = [
            PathPlanMode.A_STAR,
            PathPlanMode.BEAM_SEARCH
        ]
        if h == 1:
            print("Modes: 1. A_STAR, 2. BEAM_A_STAR")
            print("Using Manhattan heuristic")
        else:
            print("Modes: 1. A_STAR, 2. BEAM_A_STAR")
            print("Using Euclidean heuristic")
    elif h == 3 or h == 4:
        h -= 2
        modes = [
            PathPlanMode.LOCAL_SEARCH
        ]
        if h == 1:
            print("Mode: LOCAL_SEARCH")
            print("Using Manhattan heuristic")
        else:
            print("Mode: LOCAL_SEARCH")
            print("Using Euclidean heuristic")

    for mode in modes:

        search_type, path, expanded, frontier_size = None, [], [], []
        if mode == PathPlanMode.DFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "DFS"
        elif mode == PathPlanMode.BFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "BFS"
        elif mode == PathPlanMode.A_STAR:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, 0)
            search_type = "A_STAR"
        elif mode == PathPlanMode.BEAM_SEARCH:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, width)
            search_type = "BEAM_A_STAR"
        elif mode == PathPlanMode.LOCAL_SEARCH:
            path = local_search(grid, start, goal, h)
            search_type = "LOCAL_SEARCH"

        if search_type:
            print(f"Mode: {search_type}")
            path_cost = 0
            for c in path:
                path_cost += cost(grid, c)
            print(f"Path length: {len(path)}")
            print(f"Path cost: {path_cost}")
            if frontier_size:
                print(f"Number of expanded states: {len(frontier_size)}")
                print(f"Max frontier size: {max(frontier_size)}\n")
            if animate == 0 or animate == 1:
                visualize_expanded(grid, start, goal, expanded, path, animation=animate)
            else:
                visualize_path(grid, start, goal, path)
