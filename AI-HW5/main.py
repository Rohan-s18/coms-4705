import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser(
        prog="COMSW4701HW5",
        description="Grid World Localization",
    )
    parser.add_argument("-m", type=int, default=0,
                        help="Mode: 0 for forward algorithm (default), "
                             "1 for particle filter,"
                             "2 for animating forward algorithm,"
                             "3 for animating particle filter")
    parser.add_argument("-t", type=int, default=50, help="number of time steps (default 50)")
    parser.add_argument("-n", type=int, default=100, help="number of episodes for inference (default 100)")
    parser.add_argument("-e", type=float, default=0.02, help="epsilon for animation (default 0.02)")
    parser.add_argument("-p", type=int, default=30, help="number of particles for particle filter (default 30)")
    args = parser.parse_args()

    walls = [(0, 4), (0, 10), (0, 14), (1, 0), (1, 1), (1, 4), (1, 6), (1, 7), (1, 9), (1, 11), (1, 13),
             (1, 14), (1, 15), (2, 0), (2, 4), (2, 6), (2, 7), (2, 13), (2, 14), (3, 2), (3, 6), (3, 11)]
    shape = (4, 16)

    if args.m == 0 or args.m == 1:
        epsilons = [0.4, 0.2, 0.1, 0.05, 0.02]
        filtering_error = inference(shape, walls, epsilons, args.t, args.n, args.m, args.p)

        for e in range(len(epsilons)):
            plt.plot(filtering_error[e], label="e=%.2f" % epsilons[e])
        plt.legend(loc="upper right")
        plt.title("Localization error")
        plt.xlabel("Time Step")
        plt.ylabel("Error")
        plt.show()

    elif args.m == 2 or args.m == 3:
        visualize_one_run(shape, walls, args.e, args.t, args.m, args.p)

    else:
        print("Invalid mode")

if __name__ == "__main__":
    main()
