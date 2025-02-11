import os

# Define the command template
command_template = "python3 main.py {world_dir} {world_id} -e {heuristic} -b {beam_width} -a 2"

# Set parameters
world_dir = "./worlds"
world_ids = [1, 2, 3, 4]  # The four worlds
heuristics = [1, 2]  # 1: Manhattan, 2: Euclidean
beam_width = 50  # Default beam width

# Run A* and Beam Search for each world with both heuristics
for world_id in world_ids:
    for heuristic in heuristics:
        # Run A* search (beam width is ignored)
        command_a_star = command_template.format(
            world_dir=world_dir, world_id=world_id, heuristic=heuristic, beam_width=0
        )
        print(f"Running A* on World {world_id} with Heuristic {heuristic}...")
        os.system(command_a_star)

        # Run Beam Search
        command_beam = command_template.format(
            world_dir=world_dir, world_id=world_id, heuristic=heuristic, beam_width=beam_width
        )
        print(f"Running Beam Search on World {world_id} with Heuristic {heuristic}...")
        os.system(command_beam)

print("Finished running A* and Beam Search on all worlds.")
