import matplotlib.pyplot as plt
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
    workspace = initialize.initialize_workspace("single_simulation")
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
