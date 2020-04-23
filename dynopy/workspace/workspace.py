import matplotlib.pyplot as plt


class Workspace:
    def __init__(self, boundary_coordinates, obstacles, landmarks):
        self.boundary_coordinates = boundary_coordinates
        self.obstacles = obstacles
        self.landmarks = landmarks
        self.agents = []

        x_coordinates = [i[0] for i in self.boundary_coordinates]
        y_coordinates = [i[1] for i in self.boundary_coordinates]

        self.x_bounds = (min(x_coordinates), max(x_coordinates))
        self.y_bounds = (min(y_coordinates), max(y_coordinates))

    def plot(self):
        """
        Plots the environment boundaries as a black dashed line, the polygon obstacles, and the robot starting position
        and goal.
        :return: none
        """
        x_coordinates = [i[0] for i in self.boundary_coordinates]
        x_coordinates.append(self.boundary_coordinates[0][0])

        y_coordinates = [i[1] for i in self.boundary_coordinates]
        y_coordinates.append(self.boundary_coordinates[0][1])

        plt.plot(x_coordinates, y_coordinates, 'k-')

        for obstacle in self.obstacles:
            obstacle.plot()

        for landmark in self.landmarks:
            landmark.plot()

        for robot in self.agents:
            robot.plot_initial()

        x_min = self.x_bounds[0]
        x_max = self.x_bounds[1] + 1
        y_min = self.y_bounds[0]
        y_max = self.y_bounds[1] + 1

        plt.axis('equal')
        plt.xticks(range(x_min, x_max, 10))
        plt.yticks(range(y_min, y_max, 10))

    def add_agent(self, agent):
        self.agents.append(agent)
        agent.set_workspace(self)
