import csv
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp_integrate

from dynopy.dataobjects.state import GroundTruth, StateEstimate
from dynopy.dataobjects.input import Input
from dynopy.dataobjects.measurement import Measurement
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
        self.workspace = None
        self.inputs = []
        self.measurements = []
        self.perfect_measurements = []
        self.particle_set = None
        self.particle_set_list = []

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
                self.inputs.append(u)

    def initialize_particle_set(self, particle_set):
        self.particle_set = particle_set
        self.particle_set_list.append(self.particle_set)


class DifferentialDrive(TwoDimensionalRobot):
    def __init__(self, name: str, color: str, Q: np.ndarray, R: np.ndarray, state: np.ndarray, state_names: list, dt, r,
                 L):
        super().__init__(name, color, Q, R, state, state_names, dt)
        self.r = r
        self.L = L

        self.current_measurement_step = 0
        self.workspace = None
        self.inputs = []
        self.ground_truth = []
        self.measurements = []
        self.perfect_measurements = []
        self.particle_set = []
        self.particle_set_list = []

    def get_ground_truth(self):
        self.ground_truth.append(GroundTruth(0, self.state, self.state_names))

        for u in self.inputs:
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

    def get_perfect_measurements(self):
        measurement_list = []
        for state in self.ground_truth:
            measurement = self.get_predicted_measurement(state)
            measurement_list.append(measurement)

        self.perfect_measurements = measurement_list

    def simulate_noisy_measurements(self):
        noisy_measurements = []
        for meas in self.perfect_measurements:
            noisy_measurements.append(self.get_noisy_measurement(meas))

        self.measurements = noisy_measurements

    def get_noisy_measurement(self, true_measurement: Measurement):
        k = true_measurement.return_step()
        sample = np.squeeze(monte_carlo_sample(true_measurement.return_data_vector(), self.R))

        noisy_measurement = Measurement(k,
                                        sample,
                                        true_measurement.return_category_list(),
                                        true_measurement.return_name_list()
                                        )

        return noisy_measurement

    def run_prediction_update(self, x_k0, u):
        """
        given an initial state and an input, this function runs the full system dynamics prediction.
        :param x_k0: initial state [StateEstimate object]
        :param u: input [Input object]
        :return: StateEstimate object for the next time step.
        """
        k0 = x_k0.return_step()
        k1 = k0 + 1
        sol = sp_integrate.solve_ivp(self.dynamics_ode, (k0 * self.dt, k1 * self.dt), x_k0.return_data_list(),
                                     args=(u, self.L, self.r))
        x_k1 = sol.y[:, -1]
        state = StateEstimate(k1, x_k1, None, x_k0.return_state_names())
        return state

    def get_predicted_measurement(self, state):
        k = state.return_step()
        measurements = []
        measurement_values = []
        output_category = []
        source_name = []

        for landmark in self.workspace.landmarks:
            measurements.extend(landmark.return_measurement(state))

        for meas in measurements:
            measurement_values.append(meas[0])
            output_category.append(meas[1])
            source_name.append(meas[2])

        return Measurement(k, np.array(measurement_values), output_category, source_name)

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
