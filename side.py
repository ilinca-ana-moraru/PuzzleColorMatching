class Side:
    def __init__(self, value, side_type, fragment_idx):
        self.value = value.astype(int)
        self.side_type = side_type
        self.fragment_idx = fragment_idx

