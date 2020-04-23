import numpy as np


class GroundTruth:
    def __init__(self, step, state, state_names):
        """

        :param step: associated time step for the state
        :param state: 1D numpy array of state values
        :param state_names: list of ordered state names
        """
        self.step = step
        self.state = state
        self.state_names = state_names

    def return_step(self):
        return self.step

    def return_data_array(self):
        return self.state

    def return_data_list(self):
        return list(self.state)

    def return_data_vector(self):
        return np.reshape(self.state, (-1, 1))


class StateEstimate:
    def __init__(self, step, x1, x2, x3, P, state_names=None):
        self.step = step
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.P = P
        self.state_names = state_names

    def return_step(self):
        return self.step


class InformationEstimate(StateEstimate):
    def __init__(self, step, i1, i2, i3, I, state_names=None):
        super().__init__(step, i1, i2, i3, I, state_names)
