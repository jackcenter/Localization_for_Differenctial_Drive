import matplotlib.pyplot as plt


class Obstacle:
    def __init__(self, the_coordinates):
        self.vertices = the_coordinates

    def plot(self):
        """
        Plots the edges of the polygon obstacle in a 2-D represented workspace.
        :return: none
        """
        x_coordinates = [i[0] for i in self.vertices]
        x_coordinates.append(self.vertices[0][0])

        y_coordinates = [i[1] for i in self.vertices]
        y_coordinates.append(self.vertices[0][1])

        plt.plot(x_coordinates, y_coordinates)