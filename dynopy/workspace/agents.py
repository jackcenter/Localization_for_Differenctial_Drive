import csv
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp_integrate

from dynopy.dataobjects.state import GroundTruth
from dynopy.dataobjects.input import Input
from dynopy.estimationtools.tools import monte_carlo_sample


class TwoDimensionalRobot:
    def __init__(self, name: str, color: str, Q: np.ndarray, R: np.ndarray, state: np.ndarray, state_names: list, dt):
        """

        :param name:
        :param color:
        :param Q:
        :param R:
        :param state:
        :param state_names
        """
        self.name = name
        self.color = color
        self.Q = Q
        self.R = R
        self.state = state
        self.state_names = state_names
        self.dt = dt

        self.current_measurement_step = 0
        self.input_list = []
        self.measurement_list = []
        self.workspace = None

    def plot_initial(self):
        """
        Plots the position of the robot as an x
        :return: none
        """
        x_i = self.state[0]
        y_i = self.state[1]
        plt.plot(x_i, y_i, 'x', color=self.color)

    def set_workspace(self, workspace):
        self.workspace = workspace

    def return_state_array(self):
        """
        converts and returns the robot's state into a numpy array
        :return: n x 1 numpy array of current state variables
        """
        return np.array(self.state).reshape((-1, 1))

    def return_state_list(self):
        """
        converts and returns the robot's state into a numpy array
        :return: list of current state variables
        """
        return self.state

    def read_inputs(self, input_file: str):
        with open(input_file, 'r', encoding='utf8') as fin:
            reader = csv.DictReader(fin, skipinitialspace=True, delimiter=',')

            for row in reader:
                u = Input.create_from_dict(row)
                self.input_list.append(u)


class DifferentialDrive(TwoDimensionalRobot):
    def __init__(self, name: str, color: str, Q: np.ndarray, R: np.ndarray, state: np.ndarray, state_names: list, dt, r,
                 L):
        super().__init__(name, color, Q, R, state, state_names, dt)
        self.r = r
        self.L = L

        self.current_measurement_step = 0
        self.workspace = None
        self.input_list = []
        self.ground_truth = []
        self.measurement_list = []
        self.perfect_measurements = []
        self.particle_set = []
        self.particle_set_list = []

    def get_ground_truth(self):

        self.ground_truth.append(GroundTruth(0, self.state, self.state_names))

        for u in self.input_list:
            x_k0 = self.ground_truth[-1].return_data_array()
            k0 = self.ground_truth[-1].return_step()

            if u.step != k0:
                print("ERROR: ground truth to input step misalignment")

            k1 = k0 + 1
            sol = sp_integrate.solve_ivp(self.dynamics_ode, (k0 * self.dt, k1 * self.dt), x_k0,
                                         args=(u, self.L, self.r))

            x_k1 = sol.y[:, -1]
            noisy_state = monte_carlo_sample(np.reshape(x_k1, (-1, 1)), self.Q)
            self.ground_truth.append(GroundTruth(k1, np.squeeze(noisy_state), self.state_names))

    @staticmethod
    def dynamics_ode(t, x, u, L, r):
        u_r = u.u_1
        u_l = u.u_2

        x3 = x[2]

        x1_dot = r/2*(u_l + u_r)*cos(x3)
        x2_dot = r/2*(u_l + u_r)*sin(x3)
        x3_dot = r/L * (u_r - u_l)

        x_dot = np.array([x1_dot, x2_dot, x3_dot])
        return x_dot
