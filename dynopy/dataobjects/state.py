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

    def return_data_dict(self):
        data_dict = {}
        for key, val in zip(self.state_names, self.state):
            data_dict[key] = val

        return data_dict


class StateEstimate:
    def __init__(self, step, state, P, state_names):
        self.step = step
        self.state = state
        self.P = P
        self.state_names = state_names

    def return_step(self):
        return self.step

    def return_data_array(self):
        return self.state

    def return_data_list(self):
        return list(self.state)

    def return_data_vector(self):
        return np.reshape(self.state, (-1, 1))

    def return_data_dict(self):
        data_dict = {}
        for key, val in zip(self.state_names, self.state):
            data_dict[key] = val

        return data_dict

    def return_state_names(self):
        return self.state_names


class InformationEstimate(StateEstimate):
    def __init__(self, step, i, I, state_names=None):
        super().__init__(step, i, I, state_names)

    def return_information_matrix(self):
        return self.P
