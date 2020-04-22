import csv
import os

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
    base_folder = os.getcwd()
    cfg = config.get_agent_parameters(name)
    state = load_pose(pose_file)
    robot = DifferentialDrive(cfg["name"], cfg["color"], cfg["process_noise"], cfg["measurement_noise"], state)
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
    Loads initial and goal state information for a robot from a text file
    :param file: path and name to the file with robot state information
    :return: dictionaries for the initial state and goal state
    """

    with open(file, 'r', encoding='utf8') as fin:

        reader = csv.DictReader(fin, skipinitialspace=True, delimiter=',')

        raw_states = []
        for state in reader:

            temporary_state = {}
            for key, value in state.items():
                try:
                    temporary_state[key] = float(value)
                except ValueError:
                    temporary_state[key] = None

            raw_states.append(temporary_state)

    initial_state = raw_states[0]

    return initial_state
