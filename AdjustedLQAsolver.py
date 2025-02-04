import torch
import torch.nn as nn
import time
import numpy as np
from math import pi
import matplotlib.pyplot as plot


class Lqa(nn.Module):
    """
    param couplings: square symmetric numpy or torch array encoding the
                     the problem Hamiltonian
    """

    def __init__(self, lengths):
        super(Lqa, self).__init__()

        self.normal_factor = lengths.max()
        self.lengths = lengths / lengths.max()
        B = 1.3
        A = 2*np.average(self.lengths) #5 * B * self.lengths.max()
        self.couplings, self.bias = self.get_Jh(A, B) # calculating the qubo params
        self.dim = self.couplings.shape[0]
        self.energy = 0.
        self.weights = torch.zeros([self.dim])

    def Qubo(self, A, B):
        # A = A.numpy()
        N_cities = self.lengths.shape[0]
        Q = np.zeros((N_cities, N_cities, N_cities, N_cities))
        inds0 = np.arange(N_cities)
        inds1 = np.concatenate((inds0[1:], inds0[0:1]))
        lengths_np = self.lengths#.numpy()
        Q[:, inds0, :, inds1] += B * np.repeat(lengths_np.reshape(1, N_cities, N_cities), N_cities, axis=0)
        dims = np.arange(N_cities)
        Q[dims, :, dims, :] += A
        inds0 = dims.reshape(N_cities, 1).repeat(N_cities, axis=1).flatten()
        inds1 = dims.reshape(1, N_cities).repeat(N_cities, axis=0).flatten()
        Q[inds0, inds1, inds0, inds1] -= Q[inds0, inds1, inds0, inds1]
        Q[:, dims, :, dims] += A
        Q[inds1, inds0, inds1, inds0] -= Q[inds1, inds0, inds1, inds0]
        b = -np.ones((N_cities, N_cities)) * 2 * A
        return Q, b

    def get_Jh(self, A, B):
        N_cities = self.lengths.shape[0]
        Q, b = self.Qubo(A, B)
        Q = torch.tensor(Q.reshape(N_cities ** 2, N_cities ** 2), dtype=torch.float32)
        b = torch.tensor(b.reshape(N_cities ** 2), dtype=torch.float32)
        Q = 0.5 * (Q + Q.t())
        J = -0.5 * Q
        h = -0.5 * (Q.sum(1) + b)
        h = h.reshape(-1, 1)
        return J, h

    def schedule(self, i, N):
        # annealing schedule
        return i / N


    def energy_ising(self, config):
        # ising energy of a configuration
        config = config.reshape(-1, 1)
        print(config.shape[0])
        print(config.shape[1])
        print(self.bias.shape[0])
        print(self.bias.shape[1])
        return -0.5 * torch.einsum('in,ij,jn->n', config, self.couplings, config) - torch.einsum('in,ik->n', config, self.bias)


    def energy_full(self, t, g):
        # cost function value
        config = torch.tanh(self.weights) * pi / 2

        # Total energy of problem Hamiltonian
        ez = self.energy_ising(torch.sin(config))
        ex = torch.cos(config).sum()

        return (t * ez * g - (1 - t) * ex), ez

    def minimise(self,
                 opt_solution,
                 step=0.003,  # learning rate
                 N=30000,  # no of iterations
                 g=1, # balances the influence between the problem-specific Hamiltonian and a driver Hamiltonian
                 f=1.):

        self.weights = (2 * torch.rand([self.dim]) - 1) * f
        self.weights.requires_grad = True
        time0 = time.time()
        optimizer = torch.optim.Adam([self.weights], lr=step)
        cost_values = []

        for i in range(N):
            t = self.schedule(i+1, N)
            energy, cost = self.energy_full(t, g)

            optimizer.zero_grad()
            energy.backward()
            optimizer.step()

            if cost == opt_solution:
                break

            cost_values.append(cost.detach())

        self.opt_time = time.time() - time0

        route = self.get_order(torch.sign(self.weights))
        total_distance = self.calculate_total_distance(route)

        # x_axis = range(0, N)
        # plot.title("Convergence")
        # plot.plot(x_axis, cost_values)
        # plot.show()

        return route, total_distance, i

    def calculate_total_distance(self, route):
        distance = 0.0  # total distance between cities
        N_cities = route.shape[0]
        for i in range(N_cities - 1):
            distance += self.lengths[int(route[i])][int(route[i + 1])]
        distance += self.lengths[route[0]][route[N_cities - 1]]
        return distance * self.normal_factor

    def get_order(self, s_min):
        print(s_min)
        print(s_min.shape[0])
        inds_nonzero = np.nonzero((0.5 * (s_min + 1)).reshape(self.lengths.shape[0], self.lengths.shape[0]))
        inds_order = (inds_nonzero[:, 1].sort()[1])
        order = inds_nonzero[:, 0][inds_order]
        return order
