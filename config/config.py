import os
import numpy as np


def main():
    pass


def get_program_parameters(program_name):
    base_folder = os.getcwd()

    if program_name == "single_simulation":
        map_filename = "hall_map.txt"
        obstacle_filename = "hall_obstacles.txt"
        landmark_filename = "3_range_and_bearing.txt"
        pose1_filename = "pose_36_9_0.txt"
        pose2_filename = "pose_135_80_270.txt"
        inputs1_filename = "inputs1.csv"
        inputs2_filename = "inputs2.csv"

    else:
        map_filename = "hall_map.txt"
        obstacle_filename = "hall_obstacles.txt"
        landmark_filename = "3_range_and_bearing.txt"
        pose1_filename = "pose_36_9_0.txt"
        pose2_filename = "pose_135_80_270.txt"
        inputs1_filename = "inputs1.csv"
        inputs2_filename = "inputs2.csv"

    cfg = {
        "base_folder": base_folder,
        "map_file": os.path.join(base_folder, 'settings', 'maps', map_filename),
        "obstacle_file": os.path.join(base_folder, 'settings', 'obstacles', obstacle_filename),
        "landmark_file": os.path.join(base_folder, 'settings', 'landmarks', landmark_filename),
        "pose1_file": os.path.join(base_folder, 'settings', 'poses', pose1_filename),
        "pose2_file": os.path.join(base_folder, 'settings', 'poses', pose2_filename),
        "inputs1_file": os.path.join(base_folder, 'settings', 'inputs', inputs1_filename),
        "inputs2_file": os.path.join(base_folder, 'settings', 'inputs', inputs2_filename),
        "dt": 0.5,
    }

    return cfg


def get_agent_parameters(agent_name: str):

    if agent_name == "Blinky":
        color = "firebrick"
        axel_length = 10
        wheel_radius = 2.5
        Q = np.diag([.1, .1, .1, .1, .1])
        R = np.diag([5, .1, 5, .1, 5, .1])

    elif agent_name == "Pinky":
        color = "orchid"
        axel_length = 10
        wheel_radius = 2.5
        Q = np.diag([.1, .1, .1, .1, .1])
        R = np.diag([5, .1, 5, .1, 5, .1])

    elif agent_name == "Inky":
        color = "darkcyan"
        axel_length = 10
        wheel_radius = 2.5
        Q = np.diag([.1, .1, .1, .1, .1])
        R = np.diag([5, .1, 5, .1, 5, .1])

    elif agent_name == "Clyde":
        color = "darkorange"
        axel_length = 10
        wheel_radius = 2.5
        Q = np.diag([.1, .1, .1, .1, .1])
        R = np.diag([5, .1, 5, .1, 5, .1])

    else:
        print("ERROR: requested agent configuration not found. Loading Blinky")
        agent_name = "Blinky"
        color = "firebrick"
        axel_length = 10
        wheel_radius = 2.5
        Q = np.diag([.1, .1, .1, .1, .1])
        R = np.diag([5, .1, 5, .1, 5, .1])

    cfg = {
        "name": agent_name,
        "color": color,
        "axel_length": axel_length,
        "wheel_radius": wheel_radius,
        "process_noise": Q,
        "measurement_noise": R,
    }

    return cfg


if __name__ == "__main__":
    main()
