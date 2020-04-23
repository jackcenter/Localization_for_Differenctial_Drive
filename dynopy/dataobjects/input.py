class Input:
    def __init__(self, step, input_1, input_2, input_names=None):
        self.step = step
        self.u_1 = input_1
        self.u_2 = input_2
        self.input_names = input_names

    @staticmethod
    def create_from_dict(lookup):
        """
        Used to construct objects directly from a CSV data file
        :param lookup: dictionary keys
        :return: constructed ground truth object
        """
        return Input(
            int(lookup['step']),
            float(lookup['u_1']),
            float(lookup['u_2']),
        )