import csv
import matplotlib.pyplot as plt
import numpy as np


class TwoDimensionalRobot:
    def __init__(self, name: str, color: str, Q: np.ndarray, R: np.ndarray, state: dict):
        """

        :param name:
        :param color:
        :param Q:
        :param R:
        :param state:
        """
        self.name = name
        self.color = color
        self.Q = Q
        self.R = R
        self.state = state

        self.state_names = list(state.keys())
        self.current_measurement_step = 0
        self.input_list = []
        self.measurement_list = []
        self.workspace = None

    def plot_initial(self):
        """
        Plots the position of the robot as an x
        :return: none
        """
        x_i = self.state.get(self.state_names[0])
        y_i = self.state.get(self.state_names[1])
        plt.plot(x_i, y_i, 'x', color=self.color)

    def set_workspace(self, workspace):
        self.workspace = workspace

    def return_state_array(self):
        """
        converts and returns the robot's state into a numpy array
        :return: n x 1 numpy array of current state variables
        """
        state_list = list(self.state.values())
        return np.array(state_list).reshape((-1, 1))

    def return_state_list(self):
        """
        converts and returns the robot's state into a numpy array
        :return: n x 1 numpy array of current state variables
        """
        return list(self.state.values())

    def read_inputs(self, input_file: str):
        with open(input_file, 'r', encoding='utf8') as fin:
            reader = csv.DictReader(fin, skipinitialspace=True, delimiter=',')

            for row in reader:
                pass
                # u = Input.create_from_dict(row)
                # self.input_list.append(u)


class DifferentialDrive(TwoDimensionalRobot):
    def __init__(self, name: str, color: str, Q: np.ndarray, R: np.ndarray, state: dict):
        super().__init__(name, color, Q, R, state)

        self.state_names = list(state.keys())

        self.current_measurement_step = 0
        self.input_list = []
        self.measurement_list = []
        # self.ground_truth = [GroundTruth.create_from_list(0, list(state.values()))]
        self.perfect_measurements = []
        self.particle_set = []
        self.particle_set_list = []
