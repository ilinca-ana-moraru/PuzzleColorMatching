class Side:
    def __init__(self, value, grad, side_indexes_of_fragment, side_idx, fragment_idx):
        self.value = value.astype(int)
        self.grad = grad
        self.side_indexes_of_fragment = side_indexes_of_fragment
        self.side_idx = side_idx
        self.fragment_idx = fragment_idx

