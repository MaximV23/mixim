from Client import Client
from numpy.random import exponential
from Message import Message
from random import choice, sample
import numpy as np



class Mix:
    def __init__(self, id, simulation, position, n_targets, corrupt):
        # logging
        self.n_targets = n_targets
        self.id = id
        self.env = simulation.env
        self.simulation = simulation
        self.layer = position  # 1, 2, 3, ... this has no meaning in a freeroute simulation!
        self.corrupt = corrupt  # corrupt mix or not
        self.Pmix = []# probability this mix contains the target message

        for i in range(0, self.n_targets):
            self.Pmix.append(float(0.0))

    def create_dummies(self, dummy_id):
        network_dict = self.simulation.network.network_dict
        route = [None]
        delays = [0]
        for layer in range(1, self.simulation.n_layers + 1):
            if self.layer > layer:
                route.append(None)
                delays.append(0)
            elif self.layer == layer:
                route.append(self)
                delays.append(exponential(self.simulation.mu))
            else:
                route.append(choice(network_dict[layer]))
                delays.append(exponential(self.simulation.mu))
        delays += [0]
        newstop = sample(sorted(self.simulation.clientsSet, key=lambda c: c.id), k=1)[0]
        route += [newstop]
        pr_target = []
        for j in range(0, self.n_targets):
            pr_target.append(float(0.0))
        new_dummy = Message(dummy_id, 'Dummy', self, route, delays, pr_target, False)

        new_dummy.next_hop_index = self.layer + 1
        new_dummy.id = f'd_{self.id}_{dummy_id}'
        return new_dummy

    def __str__(self):
        return f'( id: {self.id}, corrupt: {self.corrupt} )'

    def __repr__(self):
        return self.__str__()


