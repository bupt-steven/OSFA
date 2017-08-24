import numpy as np

class state:
    def __init__(
            self,
            shape,
    ):
        self.demand = np.zeros(shape)
        self.topo = np.zeros(shape)

        self.demand_shape = shape
        self.topo_shape = shape

    def set_demand(self, demand):
        self.demand = demand

    def set_topo(self, topo):
        self.topo = topo
