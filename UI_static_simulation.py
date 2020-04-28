import static_simulation as sim


def main():
    working = True

    while working:
        print_header()
        cmd = get_user_input()
        working = interpret_command(cmd)

    return 0


def print_header():
    print('---------------------------------------------------')
    print('              Static Simulation ')
    print('---------------------------------------------------')
    print()


def get_user_input():
    print('This program runs the following simulations:')
    print(' [1]: Grid Approximation')
    print(' [2]: Importance Sampling')
    print(' [3]: Sequential Importance Sampling (Bootstrap) Particle Filter')
    print(' [4]: Decentralized Data Fusion')
    print(' [q]: Quit')
    print()

    cmd = input(' Select an exercise would you like to run: ')
    print()
    cmd = cmd.strip().lower()
    return cmd


def interpret_command(cmd):
    if cmd == '1':      # path planning
        print('---------------------------------------------------')
        print('              Static Simulation:')
        print('              Grid Approximation')
        print('---------------------------------------------------')
        print()
        sim.run("Grid_Approximation")

    elif cmd == '2':
        print('---------------------------------------------------')
        print('              Static Simulation:')
        print('             Importance Sampling')
        print('---------------------------------------------------')
        print()
        sim.run("Importance_Sampling")

    elif cmd == '3':
        print('---------------------------------------------------')
        print('              Static Simulation:')
        print('           Bootstrap Particle Filter')
        print('---------------------------------------------------')
        print()
        sim.run("Particle_Filter")

    elif cmd == '4':
        print('---------------------------------------------------')
        print('              Static Simulation:')
        print('           Decentralized Data Fusion')
        print('---------------------------------------------------')
        print()
        sim.run("Decentralized_Date_Fusion")

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