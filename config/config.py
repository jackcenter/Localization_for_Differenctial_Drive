import os


def main():
    pass


def get_parameters(program_name):
    base_folder = os.getcwd()

    if program_name == "single_simulation":
        map_filename = "hall_map.txt"
        obstacle_filename = "hall_obstacles.txt"
        landmark_filename = "3_range_and_bearing.txt"
        settings_filename = "firebirck.txt"
        pose_filename = "pose_36_9_0.txt"
        inputs_filename = "inputs2.csv"

    else:
        map_filename = "hall_map.txt"
        obstacle_filename = "hall_obstacles.txt"
        landmark_filename = "3_range_and_bearing.txt"
        settings_filename = "firebirck.txt"
        pose_filename = "pose_36_9_0.txt"
        inputs_filename = "inputs1.csv"

    cfg = {
        "base_folder": base_folder,
        "map_file": os.path.join(base_folder, 'settings', 'maps', map_filename),
        "obstacle_file": os.path.join(base_folder, 'settings', 'obstacles', obstacle_filename),
        "landmark_file": os.path.join(base_folder, 'settings', 'landmarks', landmark_filename),
        "settings_file": os.path.join(base_folder, 'robots', settings_filename),
        "pose_file": os.path.join(base_folder, 'robots', 'poses', pose_filename),
        "inputs_file": os.path.join(base_folder, 'robots', 'inputs', inputs_filename),
        "dt": 0.5,
    }

    return cfg


if __name__ == "__main__":
    main()
