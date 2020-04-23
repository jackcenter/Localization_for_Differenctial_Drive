import matplotlib.pyplot as plt

from config import config
from tools import initialize


def main():
    pass


def run(sim_name):
    workspace = setup()

    if sim_name == "Grid_Approximation":
        simulate_Grid_Approximation(workspace)

    elif sim_name == "Importance_Sampling":
        simulate_Importance_Sampling(workspace)

    elif sim_name == "Particle_Filter":
        simulate_Particle_Filter(workspace)


def setup():
    cfg = config.get_program_parameters("single_simulation")
    workspace = initialize.initialize_workspace("single_simulation")

    agent1 = initialize.initialize_agent("Inky", cfg["pose1_file"])
    agent1.read_inputs(cfg["inputs1_file"])
    workspace.add_agent(agent1)

    agent2 = initialize.initialize_agent("Pinky", cfg["pose2_file"])
    agent2.read_inputs(cfg["inputs2_file"])
    workspace.add_agent(agent2)

    for agent in workspace.agents:
        agent.get_ground_truth()

    workspace.plot()
    plt.show()

    return workspace


def simulate_Grid_Approximation(workspace):
    pass


def simulate_Importance_Sampling(workspace):
    pass


def simulate_Particle_Filter(workspace):
    pass


if __name__ == "__main__":
    main()
