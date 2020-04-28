import numpy as np


class Measurement:
    def __init__(self, step: int, measurement: np.ndarray, measurement_categories: list, measurement_names: list):
        """

        :param step:  associated time step for the state
        :param measurement: 1D numpy array of measurement values
        :param measurement_categories: list indicating type of measurement
        :param measurement_names: list of ordered measurement names
        """

        self.step = step
        self.measurement = measurement
        self.categories = measurement_categories
        self.names = measurement_names

    def return_step(self):
        return self.step

    def return_data_array(self):
        return self.measurement

    def return_data_list(self):
        return list(self.measurement)

    def return_data_vector(self):
        return np.reshape(self.measurement, (-1, 1))

    def return_category_list(self):
        return self.categories

    def return_name_list(self):
        return self.names
