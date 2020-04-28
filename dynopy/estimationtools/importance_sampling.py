import copy
import numpy as np
import scipy.stats as stats
from dynopy.dataobjects.state import StateEstimate


def bootstrap(particle_set: list, z_k1, agent, u):
    normalized_particle_set = SIS(particle_set, z_k1, agent, u)
    resampled_particle_set = resample(normalized_particle_set)
    return resampled_particle_set


def construct_initial_particles(distro_list, samples, Q, state_names):

    k = 0
    w = 1/samples
    particle_set = []
    for i in range(0, samples):
        state_values = []

        for distro in distro_list:
            state_values.append(distro.rvs())

        particle_set.append((StateEstimate(k, state_values, Q, state_names), w))

    return particle_set


def SIS(particle_set: list, z_k1, agent, u):
    """
    executes Sequential Importance Sampling on the agent
    :param particle_set:
    :param z_k1: measurement
    :param agent:
    :param u:
    :return: list of particles and weights for the next time step.
    """
    z_distro = stats.norm(0, np.sqrt(np.diag(agent.R).reshape(-1, 1)))
    next_particle_set = []

    for particle in particle_set:
        x_k0_m = particle[0]      # last state estimate in the particle
        w_k0 = particle[1]        # last weight in the particle
        k1 = x_k0_m.step + 1

        # Prediction:
        # define q:
        x_k0_p = agent.run_prediction_update(x_k0_m, u)
        q_distro = stats.norm(x_k0_p.return_data_array(), np.diag(agent.Q).reshape(-1, ))

        # sample q
        x_s = q_distro.rvs()
        x_s = StateEstimate(k1, x_s, None, x_k0_m.return_state_names())

        # determine the probability of z
        z_k1_hat = agent.get_predicted_measurement(x_s)
        innovation = z_k1.return_data_array() - z_k1_hat.return_data_array()
        p_z = z_distro.pdf(innovation)
        p_z = np.product(p_z)
        w_k1 = p_z*w_k0

        next_particle_set.append([x_s, w_k1])

    total_weight = sum([w[1] for w in next_particle_set])

    normalized_particle_set = []
    for particle in next_particle_set:
        normalized_particle_set.append((particle[0], particle[1]/total_weight))

    return normalized_particle_set


def resample(particle_set):
    # initialize the CSW

    csw = [0]
    for particle in particle_set:
        w_i = particle[1]
        csw.append(csw[-1] + w_i)

    csw.pop(0)

    N = len(particle_set)
    u_1 = stats.uniform(0, 1/N).rvs()

    new_particle_set = []
    for j in range(1, N+1):
        i = 0
        u_j = u_1 + (j - 1)/N

        while u_j > csw[i]:
            i += 1

        new_particle_set.append((copy.deepcopy(particle_set[i][0]), 1/N, i))

    return new_particle_set
