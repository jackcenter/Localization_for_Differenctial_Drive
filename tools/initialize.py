import csv

import numpy as np

from config import config
from dynopy.workspace.agents import DifferentialDrive
from dynopy.workspace.landmarks import Landmark
from dynopy.workspace.obstacles import Obstacle
from dynopy.workspace.workspace import Workspace


def initialize_workspace(program_name):
    """

    :param program_name:
    :return:
    """
    cfg = config.get_program_parameters(program_name)
    bounds = load_map_data(cfg["map_file"])
    obstacles = load_obstacle_data(cfg["obstacle_file"])
    landmarks = load_landmark_data(cfg["landmark_file"])
    return Workspace(bounds, obstacles, landmarks)


def initialize_agent(name, pose_file):
    """

    :param name:
    :param pose_file:
    :return:
    """
    cfg = config.get_agent_parameters(name)
    state, state_names = load_pose(pose_file)
    robot = DifferentialDrive(cfg["name"], cfg["color"], cfg["process_noise"], cfg["measurement_noise"], state,
                              state_names, cfg["dt"], cfg["wheel_radius"], cfg["axel_length"])

    return robot


def load_map_data(file):
    """

    :param file:
    :return:
    """
    environment_bounds = []

    with open(file, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin, skipinitialspace=True, delimiter=',')

        raw_bounds = next(reader)
        while raw_bounds:
            x_coordinate = int(raw_bounds.pop(0))
            y_coordinate = int(raw_bounds.pop(0))
            coordinate = (x_coordinate, y_coordinate)
            environment_bounds.append(coordinate)

    return environment_bounds


def load_obstacle_data(file):
    """

    :param file:
    :return:
    """
    obstacles = []

    with open(file, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin, skipinitialspace=True, delimiter=',')

        for raw_obstacle in reader:
            temporary_obstacle = []

            while raw_obstacle:
                x_coordinate = float(raw_obstacle.pop(0))
                y_coordinate = float(raw_obstacle.pop(0))
                coordinate = (x_coordinate, y_coordinate)
                temporary_obstacle.append(coordinate)

            obstacles.append(Obstacle(temporary_obstacle))

    return obstacles


def load_landmark_data(file):
    """

    :param file:
    :return:
    """
    with open(file, 'r', encoding='utf8') as fin:
        reader = csv.DictReader(fin, skipinitialspace=True, delimiter=',')

        landmarks = []
        for settings in reader:
            landmarks.append(Landmark.create_from_dict(settings))

    return landmarks


def load_robot_settings(file):
    """

    :param file:
    :return:
    """
    with open(file, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin, skipinitialspace=True, delimiter=',')

        keys = next(reader)
        values = next(reader)

        settings = {}
        for key, val in zip(keys, values):
            try:
                val = float(val)
            except ValueError:
                print("    Error: {} cannot be converted to a float".format(val))

            settings.update({key: val})

        return settings


def load_pose(file):
    """
    Loads initialstate information for a robot from a text file
    :param file: path and name to the file with robot state information
    :return: dictionaries for the initial state and goal state
    """

    with open(file, 'r', encoding='utf8') as fin:

        reader = csv.reader(fin, skipinitialspace=True, delimiter=',')

        state_names = next(reader)
        raw_state_list = next(reader)

        state_list = []
        for state in raw_state_list:
            state_list.append(float(state))

    return np.array(state_list), state_names
