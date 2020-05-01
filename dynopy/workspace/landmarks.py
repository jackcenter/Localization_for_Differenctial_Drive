from math import atan2
import matplotlib.pyplot as plt
import numpy as np


class Landmark:
    def __init__(self, name, x, y, model):
        self.name = name
        self.vertices = (x, y)
        self.type = model

        self.range_measurements = True
        self.bearing_measurements = True

    def plot(self):
        """
        Plots as a square
        :return: none
        """
        plt.plot(self.vertices[0], self.vertices[1], '1', markersize=12)

    def get_x(self):
        return self.vertices[0]

    def get_y(self):
        return self.vertices[1]

    def return_measurement(self, state):

        state_dict = state.return_data_dict()
        x1 = state_dict["x1"]
        y1 = state_dict["x2"]

        x2 = self.vertices[0]
        y2 = self.vertices[1]

        if self.range_measurements and self.bearing_measurements:
            r = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)
            b = atan2(y2 - y1, x2 - x1)

            return [(r, 'range', self.name), (b, 'bearing', self.name)]

        elif self.range_measurements:
            r = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)

            return [(r, 'range', self.name)]

        elif self.bearing_measurements:
            b = atan2(y2 - y1, x2 - x1)

            return [(b, 'bearing', self.name)]

        else:
            return []

    def return_grid_measurement(self, X, Y):

        x2 = self.vertices[0]
        y2 = self.vertices[1]

        if self.range_measurements and self.bearing_measurements:
            r = np.sqrt(np.square(x2 - X) + np.square(y2 - Y))
            b = np.arctan2(y2 - Y, x2 - X)

            return [(r, 'range', self.name), (b, 'bearing', self.name)]

        elif self.range_measurements:
            r = np.sqrt(np.square(x2 - X) + np.square(y2 - Y))

            return [(r, 'range', self.name)]

        elif self.bearing_measurements:
            b = np.arctan2(y2 - Y, x2 - X)

            return [(b, 'bearing', self.name)]

        else:
            return []

    def get_measurement_coordinate(self, measurement):
        if self.range_measurements and self.bearing_measurements:
            r = measurement.return_value('range', self.name)
            b = measurement.return_value('bearing', self.name)

            x = self.vertices[0] - r

        else:
            print("ERROR: landmark does not provide both range and bearing measurements so it cannot provide a "
                  "coordinate")


    @staticmethod
    def create_from_dict(settings):

        return Landmark(
            settings['name'],
            float(settings['x']),
            float(settings['y']),
            settings['model'])