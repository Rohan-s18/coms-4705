from utils.utils import *
import argparse
import path_finding as pf

def main():
    parser = argparse.ArgumentParser(
        prog="COMSW4701 HW1",
        description="Robot Path Planning",
    )
    parser.add_argument(
        "world_path", help="The directory containing worlds saved as .npy files"
    )
    parser.add_argument(
        "world_id", type=int, choices=range(1, 4), help="The world that we are testing on"
    )
    parser.add_argument(
        "-e", type=int, choices=range(0, 5), default=0, help="A* and local search heuristic options (1,3 for Manhattan; 2,4 for Euclidean)"
    )
    parser.add_argument(
        "-b", type=int, default=50, help="Beam width for beam search (default 50)"
    )
    parser.add_argument(
        "-a", type=int, choices=range(1, 3), default=0, help="Animation options (1 for expanded nodes, 2 for path)"
    )
    args = parser.parse_args()

    print("=" * 40)
    print("Testing Grid World Path Planning...")
    print(f"Loading grid world file from path: {args.world_path}")

    start_goal = [
        [(10, 10), (87, 87)],
        [(10, 10), (90, 90)],
        [(10, 10), (90, 90)],
        [(24, 24), (43, 42)]
    ]

    if 0 < args.world_id < 4:
        start, goal = start_goal[args.world_id - 1]
    else:
        start, goal = [(0, 0), (0, 0)]
    pf.test_world(
        args.world_id, start, goal, args.e, args.b, args.a, args.world_path
    )

    print("Done")
    print("=" * 40)

if __name__ == "__main__":
    main()
    