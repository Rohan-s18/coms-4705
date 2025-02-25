#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Unit tests for MCTS
"""

import numpy as np
from othello_shared import get_possible_moves, play_move, compute_utility
from mcts_ai import Node, select, expand, simulate, backprop


def main():
    initial = np.array([[2,1,0,0],
                        [0,2,2,2],
                        [0,1,2,0],
                        [1,2,1,0]])

    child1 = np.array([[2,1,1,0],
                       [0,2,1,2],
                       [0,1,1,0],
                       [1,2,1,0]])

    child2 = np.array([[2,1,0,1],
                       [0,2,1,2],
                       [0,1,2,0],
                       [1,2,1,0]])

    child3 = np.array([[2,1,0,0],
                       [0,2,1,2],
                       [0,1,1,1],
                       [1,2,1,0]])

    grandchild1 = np.array([[2,2,2,0],
                            [0,2,1,2],
                            [0,1,1,1],
                            [1,2,1,0]])

    root = Node(initial, 1, None, [], 0, 20)
    node1 = Node(child1, 2, root, [], -1, 1)
    node2 = Node(child2, 2, root, [], 1.5, 17)
    node3 = Node(child3, 2, root, [], 1, 2)
    node4 = Node(grandchild1, 1, node3, [], 0, 2)
    root.children = root.children + [node1, node2, node3]
    node3.children.append(node4)

    if (select(root, 1).state == child3).all():
        print("select() test passed")
    else:
        print("select() test failed")

    grandchild2 = np.array([[2,1,0,0],
                            [0,2,1,2],
                            [0,1,2,2],
                            [1,2,2,2]])

    if (expand(node3).state == grandchild2).all():
        print("expand() test passed")
    else:
        print("expand() test failed")

    node5 = Node(grandchild2, 1, node3, [], 0, 0)
    node3.children.append(node5)

    res = simulate(node5)
    if res == -6 or res == -10:
        print("simulate() test passed")
    else:
        print("simulate() test failed")

    backprop(node5, -6)
    if node5.value == 6 and node3.value == -5 and root.value == 6:
        print("backprop() test passed")
    else:
        print("backprop() test failed")
        print(f"node5 value: {node5.value} vs expected: 6")
        print(f"node3 value: {node3.value} vs expected: -5")
        print(f"root value: {root.value} vs expected: 6")


if __name__ == "__main__":
    main()
