import torch
import torch.nn as nn
import time
import numpy as np
from math import pi
import matplotlib.pyplot as plot
from pyqubo import Array, Constraint



class Lqa(nn.Module):
    """
    param couplings: square symmetric numpy or torch array encoding the
                     the problem Hamiltonian
    """

    def __init__(self, couplings):
        super(Lqa, self).__init__()

        self.lengths = torch.from_numpy(couplings).float()
        self.n = couplings.shape[0]
        self.energy = 0.
        self.config = torch.zeros([self.n, self.n])
        self.min_en = 9999.
        self.min_config = torch.zeros([self.n, self.n])
        self.weights = torch.zeros([self.n, self.n])

    def schedule(self, i, N):
        # annealing schedule
        return i / N


    def dist(self, i, j, cities):
        pos_i = np.array(cities[i][1])
        pos_j = np.array(cities[j][1])
        return np.linalg.norm(np.subtract(pos_i, pos_j))  # Euclidean distance


    def alpha(self, n, h, maxdist):
        return ((((n ** 3) - (2 * n ** 2) + n) * maxdist) + h)


    def beta(self, n, alfa, h):
        return ((((n ** 2) + 1) * alfa) + h)


    def exp1(self, n, v, A):
        exp = 0.0
        for i in range(n):
            for j in range(n):
                exp += Constraint(1 - v[i, j], label="exp")
        exp = A * exp
        return (exp)


    def exp2(self, n, v, B):
        exp = 0.0
        for i in range(n):
            for j in range(n):
                if (i != j):
                    for k in range(n):
                        exp += Constraint(((v[i, k]) * (v[j, k])), label="exp")
        exp = B * exp
        return (exp)


    def exp3(self, n, v, B):
        exp = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (k != j):
                        exp += Constraint((v[i, j]) * (v[i, k]), label="exp")
        exp = B * exp
        return (exp)


    def exp4(self, n, v):
        exp = 0.0
        cities = list(tuple((i, self.couplings_tensor[i]) for i in range(self.n)))
        for i in range(n):
            for j in range(n):
                if (i != j):
                    for k in range(n - 1):
                        exp += self.dist(i, j, cities) * Constraint(((v[i, k] * v[j, k + 1])), label="exp")
        return (exp)


    def Qubo(self):
        h = 0.0000005  # small number

        """Prepare binary vector with  bit $(i, j)$ representing to visit $j$ city at time $i$"""

        v = Array.create('v', (self.n, self.n), 'BINARY')
        A = self.alpha(self.n, h, torch.max(torch.abs(self.couplings_tensor)))
        B = self.beta(self.n, A, h)

        # Define hamiltonian H
        H = self.exp1(self.n, v, A) + self.exp2(self.n, v, B) + self.exp3(self.n, v, B) + self.exp4(self.n, v)

        # Compile model
        model = H.compile()

        # Create QUBO
        qubo, offset = model.to_qubo()

        return qubo, offset


    def energy_ising(self, config):
        # ising energy of a configuration
        xmat = self.to_binary_matrix(config).float()
        perm = self.perm(self.n).float()
        travel_cost = xmat.t().mm(self.couplings).mm(xmat).mm(perm).trace() / 2
        route, travel_cost = self.calculate_total_distance()
        return travel_cost


    def energy_full(self, t, g, A):
        matrix = self.complete_matrix()

        # cost function value
        config = torch.tanh(matrix) * pi / 2

        # Total energy of problem Hamiltonian
        ez = self.energy_ising(torch.sin(config)) + self.caculate_bias() + A * self.penalty(torch.sin(config))
        ex = torch.cos(config).sum()

        return (t * ez * g - (1 - t) * ex), ez


    def minimise(self,
                 opt_solution,
                 step=0.01,  # learning rate
                 N=2000,  # no of iterations
                 g=1, # balances the influence between the problem-specific Hamiltonian and a driver Hamiltonian
                 f=1.,
                 A=100): # penalty

        qubo, offset = self.Qubo

        self.weights = (2 * torch.rand([qubo.shape[0]]) - 1) * f
        self.weights.requires_grad = True
        time0 = time.time()
        optimizer = torch.optim.Adam([self.weights], lr=step)
        distances = []

        for i in range(N):
            t = self.schedule(i+1, N)
            energy, cost = self.energy_full(t, g, A)

            optimizer.zero_grad()
            energy.backward()
            optimizer.step()

            route, total_distance = self.calculate_total_distance()
            #print(f"route {route}, distance {total_distance}, ising energy {self.energy_ising(self.complete_matrix())}, full energy {energy}")

            distances.append(abs(total_distance - opt_solution))

            if cost == opt_solution:
                break

        self.opt_time = time.time() - time0

        self.print_solution()

        x_axis = range(0, N)
        plot.title("Convergence")
        plot.plot(x_axis, distances)
        plot.show()

        return route, total_distance, i

    def caculate_bias(self):
        matrix = torch.tanh(self.complete_matrix())

        _, indices = torch.max(matrix, dim=0)
        binary_matrix = torch.zeros_like(matrix)
        binary_matrix[indices, torch.arange(matrix.shape[1])] = 1

        bias_tensor = self.bias.view(-1, 1)
        biased_matrix = binary_matrix * bias_tensor

        # Sum all elements of the biased_matrix
        total_sum = torch.sum(biased_matrix)

        return total_sum


    def calculate_total_distance(self):
        matrix = self.complete_matrix()
        # Calculate the tour
        x_final = torch.tanh(matrix) * pi / 2

        route = [0]
        for j in range(1, self.n):
            i = torch.argmax(x_final[:, j]).item()
            route.append(i)
        route.append(0)  # return to the start city

        total_distance = 0
        for i in range(self.n):
            total_distance += self.couplings[route[i]][route[i + 1]]

        total_distance *= self.normal_factor
        return route, total_distance

    def print_solution(self):
        route, total_distance = self.calculate_total_distance()

        print("Route: ", route)
        print("Total distance: ", total_distance)
