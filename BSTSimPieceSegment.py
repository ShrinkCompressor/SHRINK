class BSTSimPieceSegment:
    def __init__(self, init_timestamp, a_min, a_max, b=None):
        self.init_timestamp = init_timestamp
        self.a_min = a_min
        self.a_max = a_max
        self.b = b
        
        # Add the 'right' attribute
        self.right = None
        self.left = None

    # If you need getter methods similar to Java, you can use properties in Python.
    @property
    def get_init_timestamp(self):
        return self.init_timestamp

    @property
    def get_a_min(self):
        return self.a_min

    @property
    def get_a_max(self):
        return self.a_max

    @property
    def get_a(self):
        return (self.a_min + self.a_max) / 2

    @property
    def get_b(self):
        return self.b