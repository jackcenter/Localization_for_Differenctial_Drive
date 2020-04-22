import csv
import matplotlib.pyplot as plt
import numpy as np


class TwoDimensionalRobot:
    def __init__(self, settings: dict, state: dict):
        self.settings = settings
        self.state = state

        self.name = settings.get('name')
        self.color = settings.get('color')
        self.state_names = list(state.keys())

        self.current_measurement_step = 0
        self.input_list = []
        self.measurement_list = []

    def plot_initial(self):
        """
        Plots the position of the robot as an x
        :return: none
        """
        x_i = self.state.get(self.state_names[0])
        y_i = self.state.get(self.state_names[1])
        plt.plot(x_i, y_i, 'x', color=self.color)

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
                u = Input.create_from_dict(row)
                self.input_list.append(u)


class DifferentialDrive(TwoDimensionalRobot):
    def __init__(self, settings: dict, state: dict):
        super().__init__(settings, state)

        self.name = settings.get('name')
        self.color = settings.get('color')
        self.axel_length = settings.get('axel_length')
        self.wheel_radius = settings.get('wheel_radius')
        self.state_names = list(state.keys())

        self.workspace = None
        self.Q = np.array([[]])
        self.R = np.array([[]])

        self.current_measurement_step = 0
        self.input_list = []
        self.measurement_list = []
        self.ground_truth = [GroundTruth.create_from_list(0, list(state.values()))]
        self.perfect_measurements = []
        self.particle_set = []
        self.particle_set_list = []
