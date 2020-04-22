import single_simulation as sim


def main():
    working = True

    while working:
        print_header()
        cmd = get_user_input()
        working = interpret_command(cmd)

    return 0


def print_header():
    print('---------------------------------------------------')
    print('              Individual Simulation ')
    print('---------------------------------------------------')
    print()


def get_user_input():
    print('This program runs the following simulations:')
    print(' [1]: Grid Approximation')
    print(' [2]: Importance Sampling')
    print(' [3]: Sequential Importance Sampling (Bootstrap) Particle Filter')
    print(' [q]: Quit')
    print()
    print(' NOTE: parameters for landmarks and dynamics models can be changed in settings.')
    print()

    cmd = input(' Select an exercise would you like to run: ')
    cmd = cmd.strip().lower()
    return cmd


def interpret_command(cmd):
    if cmd == '1':      # path planning
        sim.run("Grid_Approximation")

    elif cmd == '2':
        sim.run("Importance_Sampling")

    elif cmd == '3':
        sim.run("Particle_Filter")

    elif cmd == 'q':
        return False

    else:
        print(' ERROR: unexpected command...')

    print()
    run_again = input(' Would you like to run another exercise?[y/n]: ')

    if run_again != 'y':
        return False

    return True


if __name__ == '__main__':
    main()