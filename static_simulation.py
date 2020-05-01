from math import pi

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from config import config
from dynopy.estimationtools.importance_sampling import construct_initial_particles, SIS, bootstrap
from tools import initialize


def main():
    pass


def run(sim_name):

    k_final = 5
    particles = 10
    if sim_name == "Grid_Approximation":
        workspace = setup(1)
        x1, y1, m1, x2, y2, m2, x3, y3, m3, post = simulate_Grid_Approximation(workspace, k_final)
        print("MMSE estimate:\n  x =     {}\n  y =     {}\n  theta = {}".format(m1, m2, m3))
        plot_grid_approximation(x1, y1, m1, x2, y2, m2, x3, y3, m3)

        plt.figure()
        plt.imshow(np.sum(post, axis=2), cmap=plt.cm.Greys, origin='lower', extent=[0, 100, 41, 59])
        workspace.plot()

    elif sim_name == "Importance_Sampling":
        workspace = setup(1)
        agent = simulate_Importance_Sampling(workspace, particles, k_final)

        plot_particles(agent.particle_set_list, 0)
        plot_particles(agent.particle_set_list, -1)

    elif sim_name == "Particle_Filter":
        workspace = setup(1)
        agent = simulate_Particle_Filter(workspace, particles, k_final)

        plot_particles(agent.particle_set_list, 0)

    elif sim_name == "Decentralized_Date_Fusion":
        workspace = setup(2)
        simulate_DDF(workspace)

    elif sim_name == "Comparison":
        workspace = setup(2)
        x1, y1, m1, x2, y2, m2, x3, y3, m3, post = simulate_Grid_Approximation(workspace, k_final)
        agent = simulate_Particle_Filter(workspace, particles, k_final)
        plot_comparison(x1, y1, m1, x2, y2, m2, x3, y3, m3, agent.particle_set_list, 0)
        plot_comparison(x1, y1, m1, x2, y2, m2, x3, y3, m3, agent.particle_set_list)
        plt.figure()
        workspace.plot()

    plt.show()
    return 0


def setup(number_of_agents):
    """
    Creates the appropriate workspace with bounds, obstacles, and landmarks as well as initializing the robot for the
    simulation. This includes creating a truth model for the robot based on the inputs.
    :return: workspace object
    """
    cfg = config.get_program_parameters("static_simulation")
    workspace = initialize.initialize_workspace("static_simulation")

    if number_of_agents == 1:
        agent1 = initialize.initialize_agent("Blinky", cfg["pose1_file"])
        agent1.read_inputs(cfg["inputs1_file"])
        workspace.add_agent(agent1)

    elif number_of_agents == 2:
        agent1 = initialize.initialize_agent("Blinky", cfg["pose1_file"])
        agent1.read_inputs(cfg["inputs1_file"])
        workspace.add_agent(agent1)

        agent2 = initialize.initialize_agent("Inky", cfg["pose2_file"])
        agent2.read_inputs(cfg["inputs2_file"])
        workspace.add_agent(agent2)

    for agent in workspace.agents:
        agent.get_ground_truth()
        agent.get_perfect_measurements()
        agent.simulate_noisy_measurements()

    # workspace.plot()
    # plt.show()

    return workspace


def simulate_Grid_Approximation(workspace, k_final=None):
    """
    runs a grid approximation for each agents
    :param workspace: workspace object created in setup
    :param k_final: last time step to estimate through
    :return: None
    """
    agent = workspace.agents[0]

    if not k_final:
        k_final = len(agent.measurements)

    if k_final > len(agent.measurements):
        print("ERROR: there are not that many measurements")
        k_final = len(agent.measurements)

    dx = 1
    dy = 1
    dtheta = pi/36

    x_bounds = workspace.x_bounds
    theta_bounds = (0, 2*pi)

    x_range = np.arange(x_bounds[0], x_bounds[1] + dx, dx)
    y_range = np.arange(41, 59 + dy, dy)
    theta_range = np.arange(theta_bounds[0], theta_bounds[1], dtheta)

    X, Y, Theta = np.meshgrid(x_range, y_range, theta_range)
    XYTheta = np.concatenate((np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1)), np.reshape(Theta, (-1, 1))), axis=1)

    # Initial probability
    x_uniform = stats.uniform(loc=5, scale=8)           # U[5, 13]
    y_uniform = stats.uniform(loc=46, scale=8)          # U[46, 54]
    theta_uniform = stats.uniform(loc=0, scale=2*pi)    # U[0, 2pi]

    pX = x_uniform.pdf(X)
    pY = y_uniform.pdf(Y)
    pTheta = theta_uniform.pdf(Theta)
    pJoint = pX*pY*pTheta

    std = np.sqrt(np.diag(agent.R))
    mu = np.zeros(len(std))

    # list of normal distributions for each measurement
    z_norm_list = []
    for m, s in zip(mu, std):
        z_norm_list.append(stats.norm(loc=m, scale=s))

    # loop through the measurements
    pPost = pJoint
    for meas in agent.measurements[0:k_final]:
        z = meas.return_data_array()

        measurements = []
        for landmark in workspace.landmarks:
            measurements.extend(landmark.return_grid_measurement(X, Y))

        ino_values = []
        for z, meas in zip(z, measurements):
            z_hat = meas[0]
            z_ino = z - z_hat
            ino_values.append(z_ino)

        pZ = np.ones(X.shape)
        for z_norm, grid in zip(z_norm_list, ino_values):
            pZ_i = z_norm.pdf(grid)
            pZ = pZ_i*pZ

        pZ = pZ/np.sum(pZ)

        pNum = pZ*pPost
        pDen = np.sum(pNum)
        pPost = pNum/pDen

    # marginal probabilities
    pX = np.sum(pPost, axis=0)
    pX = np.sum(pX, axis=1)
    X_mmse = np.trapz(x_range*pX)

    pY = np.sum(pPost, axis=1)
    pY = np.sum(pY, axis=1)
    Y_mmse = np.trapz(y_range*pY)

    pTheta = np.sum(pPost, axis=0)
    pTheta = np.sum(pTheta, axis=0)
    Theta_mmse = np.trapz(theta_range*pTheta)

    return x_range, pX, X_mmse, y_range, pY, Y_mmse, theta_range, pTheta, Theta_mmse, pPost


def simulate_Importance_Sampling(workspace, num_of_particles=10, k_final=None):
    agent = workspace.agents[0]

    if not k_final:
        k_final = len(agent.measurements)

    if k_final > len(agent.measurements):
        print("ERROR: there are not that many measurements")
        k_final = len(agent.measurements)
    # Initial probability
    x_uniform = stats.uniform(loc=5, scale=8)  # U[5, 13]
    y_uniform = stats.uniform(loc=46, scale=8)  # U[46, 54]
    theta_uniform = stats.uniform(loc=0, scale=2 * pi)  # U[0, 2pi]
    distro_list = [x_uniform, y_uniform, theta_uniform]

    state_names = agent.state_names
    initial_particle_set = construct_initial_particles(distro_list, num_of_particles, agent.Q, state_names)
    agent.initialize_particle_set(initial_particle_set)

    inputs = agent.inputs
    measurements = agent.measurements
    print(len(inputs))
    print(len(measurements))
    for u, z in zip(inputs[0: k_final], measurements[0: k_final]):
        agent.particle_set = SIS(agent.particle_set, z, agent, u)
        agent.particle_set_list.append(agent.particle_set)

    for particle in agent.particle_set_list[-1]:
        print(particle[0].__dict__, particle[1])

    print(len(agent.particle_set_list))

    return agent


def simulate_Particle_Filter(workspace, num_of_particles=10, k_final=None):
    agent = workspace.agents[0]

    if not k_final:
        k_final = len(agent.measurements)

    if k_final > len(agent.measurements):
        print("ERROR: there are not that many measurements")
        k_final = len(agent.measurements)

    x_uniform = stats.uniform(loc=5, scale=8)  # U[5, 13]
    y_uniform = stats.uniform(loc=46, scale=8)  # U[46, 54]
    theta_uniform = stats.uniform(loc=0, scale=2 * pi)  # U[0, 2pi]
    distro_list = [x_uniform, y_uniform, theta_uniform]

    state_names = agent.state_names
    initial_particle_set = construct_initial_particles(distro_list, num_of_particles, agent.Q, state_names)
    agent.initialize_particle_set(initial_particle_set)

    inputs = agent.inputs
    measurements = agent.measurements
    for u, z in zip(inputs[0: k_final], measurements[0: k_final]):
        agent.particle_set = bootstrap(agent.particle_set, z, agent, u)
        agent.particle_set_list.append(agent.particle_set)

    for particle in agent.particle_set_list[-1]:
        print(particle[0].__dict__)

    print(len(agent.particle_set_list))

    return agent


def simulate_DDF(workspace):

    print("You DDF'ed")


def plot_grid_approximation(x1, y1, m1, x2, y2, m2, x3, y3, m3):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.tight_layout(pad=1)
    fig.subplots_adjust(left=0.1)

    ax1.plot(x1, y1)
    ax1.axvline(x=m1, ymin=0.05, ymax=0.95, c='k', ls='--')
    ax1.set_xlabel(r"Easting, $\xi$  [inches]")
    ax1.set_ylabel(r"$p(\xi)$")
    ax1.grid(True)

    ax2.plot(x2, y2)
    ax2.axvline(x=m2, ymin=0.05, ymax=0.95, c='k', ls='--')
    ax2.set_xlabel(r"Northing, $\eta$ [inches]")
    ax2.set_ylabel(r"$p(\eta)$")
    ax1.grid(True)

    ax3.plot(x3, y3)
    ax3.axvline(x=m3, ymin=0.05, ymax=0.95, c='k', ls='--')
    ax3.set_xlabel(r"Orientation, $\theta$ [radians]")
    ax3.set_ylabel(r"$p(\theta)$")
    ax1.grid(True)


def plot_particles(particle_set_list, k=-1, marker='ko'):
    x1_ord = []
    x2_ord = []
    x3_ord = []
    w_ord = []

    for p in particle_set_list[k]:
        x1_ord.append(p[0].state[0])
        x2_ord.append(p[0].state[1])
        x3_ord.append(p[0].state[2])
        w_ord.append(p[1])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.tight_layout(pad=1)
    fig.subplots_adjust(left=0.1)

    ax1.plot(x1_ord, w_ord, marker)
    ax1.set_xlabel(r"Easting, $\xi$  [inches]")
    ax1.set_ylabel(r"$p(\xi)$")
    ax1.grid(True)

    ax2.plot(x2_ord, w_ord, marker)
    ax2.set_xlabel(r"Northing, $\eta$ [inches]")
    ax2.set_ylabel(r"$p(\eta)$")
    ax1.grid(True)

    ax3.plot(x3_ord, w_ord, marker)
    ax3.set_xlabel(r"Orientation, $\theta$ [radians]")
    ax3.set_ylabel(r"$p(\theta)$")
    ax1.grid(True)


def plot_comparison(x1, y1, m1, x2, y2, m2, x3, y3, m3, particle_set_list, k=-1):

    x1_ord = []
    x2_ord = []
    x3_ord = []
    w_ord = []

    for p in particle_set_list[k]:
        x1_ord.append(p[0].state[0])
        x2_ord.append(p[0].state[1])
        x3_ord.append(p[0].state[2])
        w_ord.append(p[1])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.tight_layout(pad=1)
    fig.subplots_adjust(left=0.1)
    marker = 'ko'

    ax1.plot(x1, y1)
    ax1.axvline(x=m1, ymin=0.05, ymax=0.95, c='k', ls='--')
    ax1.plot(x1_ord, w_ord, marker)
    ax1.set_xlabel(r"Easting, $\xi$  [inches]")
    ax1.set_ylabel(r"$p(\xi)$")
    ax1.grid(True)

    ax2.plot(x2, y2)
    ax2.axvline(x=m2, ymin=0.05, ymax=0.95, c='k', ls='--')
    ax2.plot(x2_ord, w_ord, marker)
    ax2.set_xlabel(r"Northing, $\eta$ [inches]")
    ax2.set_ylabel(r"$p(\eta)$")
    ax2.grid(True)

    ax3.plot(x3, y3)
    ax3.axvline(x=m3, ymin=0.05, ymax=0.95, c='k', ls='--')
    ax3.plot(x3_ord, w_ord, marker)
    ax3.set_xlabel(r"Orientation, $\theta$ [radians]")
    ax3.set_ylabel(r"$p(\theta)$")
    ax3.grid(True)


if __name__ == "__main__":
    main()
