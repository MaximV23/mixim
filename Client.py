from random import choice, sample
from Message import Message
from numpy.random import exponential
import numpy as np
import random

class Client:
    def __init__(self, simulation, id, network_dict, rate_client, mu, probability_dist_mixes, n_targets, client_dummies, rate_client_dummies, Log):
        self.id = id
        self.env = simulation.env
        self.simulation = simulation  # simulation object
        self.network_dict = network_dict
        self.class_ends = self.env.event()
        self.mu = mu  # Avg delay: for delays at the poisson mixes
        self.probability_dist_mixes = probability_dist_mixes
        self.other_clients = set()
        self.message_id = 1

        self.rate_client = rate_client
        self.all_mixes = []
        self.n_targets = n_targets
        self.client_dummies = client_dummies
        self.rate_client_dummies = rate_client_dummies
        self.log = Log
        if self.simulation.topology == 'stratified':
            for layer in range(1, len(self.network_dict) + 1):
                self.all_mixes += self.network_dict[layer]
        elif self.simulation.topology == 'XRD':
            self.set_chains = self.network_dict
        self.env.process(self.send_message('Real', self.rate_client))
        if self.client_dummies:
            self.env.process(self.send_message('ClientDummy', self.rate_client_dummies))

    def create_message(self, message_type, rate_client):
        np.random.seed()
        delay_client = exponential(rate_client)
        route = [self]
        route_ids = [self.id]
        delays = [delay_client]
        pr_target = []

        for i in range(0, self.n_targets):
            pr_target.append(float(0.0))
        for layer in range(1, self.simulation.n_layers+1):
            delay_per_mix = exponential(self.mu)
            delays.append(delay_per_mix)
            if self.simulation.routing == 'source' and self.simulation.topology == 'stratified'\
                    or (self.simulation.routing == 'hopbyhop' and self.simulation.topology == 'stratified' and layer == 1):
                if self.simulation.n_layers ==1 and self.simulation.n_mixes_per_layer == 1:
                    node = self.network_dict[1][0]
                    route.append(node)
                    route_ids.append(node.id)
                else:
                    if layer == 1:
                        node = np.random.choice(self.network_dict[layer], p=self.probability_dist_mixes[layer - 1])
                        route.append(node)
                        route_ids.append(node.id)
                    else:
                        node = np.random.choice(node.neighbors)
                        route.append(node)
                        route_ids.append(node.id)


            elif self.simulation.routing == 'source' and self.simulation.topology == 'freeroute'\
                    or (self.simulation.routing == 'hopbyhop' and self.simulation.topology == 'freeroute' and layer == 1):

                node = choice(self.all_mixes)
                while node in route:
                    node = choice(self.all_mixes)
                route.append(node)
                route_ids.append(node.id)

            elif self.simulation.routing == 'hopbyhop' and layer != 1:
                route.append(None)
                route_ids.append(None)
            elif self.simulation.routing == 'source' and self.simulation.topology == 'XRD':
                chain = random.choice(self.set_chains)
                route = [self]
                route_ids = [self.id]
                for node in chain:
                    route.append(node)
                    route_ids.append((node.id))
        delays += [0]
        receiver = sample(sorted(self.other_clients, key=lambda c: c.id), k=1)[0]
        route += [receiver]
        route_ids += [receiver.id]

        message = Message(self.message_id, message_type, self, route, delays, pr_target,False)
        if self.message_id == 1 and self.id ==1:
            for i in range(len(self.probability_dist_mixes)):
                if self.simulation.printing:
                    print("Weights Layer %d %s"%(i, self.probability_dist_mixes))
                else:
                    pass
        self.message_id += 1
        return message, delay_client



    def receive_message(self, message):
        message.timeReceived = self.env.now
        self.log.received_messages_f(message)
        if message.target_bool and self.simulation.printing:
            print(f'Target message arrived at destination Client at time {self.env.now}')
        if message.type == 'Real' or message.type == 'ClientDummy':
            message.route[0].receive_ack(message)


    def send_message(self, message_type, rate_client):
        while True:
            message, sending_time = self.create_message(message_type, rate_client)
            yield self.env.timeout(sending_time)
            message.time_left = self.env.now
            self.log.sent_messages_f(message)
            self.env.process(self.simulation.attacker.relay(message, message.route[1]))

    def receive_ack(self, message):  # Message received
        pass

    def __str__(self):
        return 'Client id: {}'.format(self.id)

    def __repr__(self):
        return self.__str__()