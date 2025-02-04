import torch
import torch.nn as nn
import time
import numpy as np
from math import pi
import matplotlib.pyplot as plot
import cProfile
import pstats


class Lqa(nn.Module):
    """
    Quantum-inspired solver for the Traveling Salesman Problem (TSP)
    using Local Quantum Annealing (LQA).
    """

    def __init__(self, couplings):
        """
        Initialize the LQA solver with the problem couplings.

        Args:
            couplings (numpy.ndarray): Square symmetric matrix encoding the problem Hamiltonian.
        """
        super(Lqa, self).__init__()

        couplings_tensor = torch.from_numpy(couplings).float() #couplings
        self.normal_factor = torch.max(torch.abs(couplings_tensor))
        self.couplings = couplings_tensor / self.normal_factor
        self.bias = torch.sum(self.couplings, dim=1) / 4
        self.n = couplings.shape[0]
        self.energy = 0.
        self.config = torch.zeros([self.n, self.n])
        self.min_en = 9999.
        self.min_config = torch.zeros([self.n, self.n])
        self.weights = torch.zeros([self.n, self.n])

    def schedule(self, i, N):
        """
        Compute the annealing schedule based on iteration progress.

        Args:
            i (int): Current iteration.
            N (int): Total number of iterations.

        Returns:
            float: Annealing schedule parameter.
        """
        # annealing schedule
        return i / N

    def perm(self, n):
        res = np.zeros([n, n])
        for i in range(n):
            res[(i + 2) % n - 1, i] = 1
        return torch.tensor(res).float()

    def to_binary_matrix(self, matrix):
        """
        Convert the solution matrix to a binary form.

        Args:
            matrix (torch.Tensor): Solution matrix.

        Returns:
            torch.Tensor: Converted binary matrix.
        """
        # Ensure matrix is a PyTorch tensor
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix)

        # Use PyTorch's argmax, no need to convert to NumPy
        max_indices = torch.argmax(matrix, axis=0)

        # Initialize a binary matrix of the same shape as the original, filled with zeros
        binary_matrix = torch.zeros_like(matrix, dtype=torch.int)

        # Set the maximum values in each column to 1
        for col_idx in range(matrix.shape[1]):
            binary_matrix[max_indices[col_idx], col_idx] = 1

        return binary_matrix

    def complete_matrix(self):
        """
        Complete a solution to make sure there is a tour.

        Returns:
            torch.Tensor: Completed matrix.
        """
        # Create a fixed first row (1 followed by zeros)
        fixed_row = torch.tensor([[1] + [0] * (self.n - 1)], dtype=torch.float32)

        # Create a fixed first column (1 followed by zeros, reshaped to a column)
        fixed_col = torch.tensor([[1] + [0] * (self.n - 1)], dtype=torch.float32).T

        # Construct the full matrix for energy calculation
        full_matrix = torch.cat((fixed_row, torch.cat((fixed_col[1:], self.weights), dim=1)), dim=0)

        return full_matrix

    def energy_ising(self, config):
        """
        Compute the Ising energy of a given configuration.

        Args:
            config (torch.Tensor): Solution vector.

        Returns:
            torch.Tensor: Computed Ising energy.
        """
        # ising energy of a configuration
        # xmat = self.to_binary_matrix(config).float()
        # perm = self.perm(self.n).float()
        # travel_cost = xmat.t().mm(self.couplings).mm(xmat).mm(perm).trace() / 2
        route, travel_cost = self.calculate_total_distance()
        return travel_cost

    def penalty(self, config):
        """
        Compute the penalty of a given configuration.

        Args:
            config (torch.Tensor): Solution vector.

        Returns:
            torch.Tensor: Computed penalty.
        """
        #xmat = self.to_binary_matrix(config)
        p = (1 - config.sum(0)).pow(2).sum() + (1 - config.sum(1)).pow(2).sum()
        return p

    def energy_full(self, t, g, A):
        """
         Compute the full cost function value combining problem Hamiltonian and driver Hamiltonian.

         Args:
             t (float): Annealing schedule parameter.
             g (float): Scaling factor for quantum tunneling.
             A (float): Penalty parameter.

         Returns:
             tuple: (Problem energy, total energy including driver Hamiltonian contribution).
         """
        matrix = self.complete_matrix()

        # cost function value
        config = torch.tanh(matrix) * pi / 2

        # Total energy of problem Hamiltonian
        ez = self.energy_ising(torch.sin(config)) + self.calculate_bias() + A * self.penalty(torch.sin(config))
        ex = torch.cos(config).sum()

        return (t * ez * g - (1 - t) * ex), ez

    def minimise(self,
                 opt_solution,
                 step=0.01,  # learning rate
                 N=1000,  # no of iterations
                 g=1, # balances the influence between the problem-specific Hamiltonian and a driver Hamiltonian
                 f=1.,
                 A=100): # penalty
        """
        Perform quantum-inspired annealing to minimise the TSP problem.

        Args:
            opt_solution (float): Optimal solution for comparison.
            step (float, optional): Learning rate for optimization. Default is 0.01.
            N (int, optional): Number of iterations. Default is 1000.
            g (float, optional): Balances the influence between the problem specific and the driver
            Hamiltonians. Default is 1.
            f (float, optional): Scaling factor for weight initialization. Default is 1.
            A (float, optional): Penalty parameter to enforce TSP constraints

        Returns:
            tuple: Best solution, its energy, convergence step, and error values over iterations.
        """

        self.weights = (2 * torch.rand([self.n - 1, self.n - 1]) - 1) * f
        self.weights.requires_grad = True
        time0 = time.time()
        optimizer = torch.optim.Adam([self.weights], lr=step)
        distances = []


        # profiler = cProfile.Profile()
        # profiler.enable()

        for i in range(N):
            t = self.schedule(i+1, N)
            energy, cost = self.energy_full(t, g, A)

            optimizer.zero_grad()
            energy.backward()
            optimizer.step()

            route, total_distance = self.calculate_total_distance()
            #print(f"route {route}, distance {total_distance}, ising energy {self.energy_ising(self.complete_matrix())}, full energy {energy}")

            distances.append(abs(total_distance - opt_solution)/opt_solution)

            if cost == opt_solution:
                break
        #profiler.disable()

        # Print out the stats
        #stats = pstats.Stats(profiler)
        #stats.sort_stats('cumtime').print_stats(10)  # Sort by cumulative time and show top 10 functions

        self.opt_time = time.time() - time0

        self.print_solution()

        x_axis = range(0, N)
        plot.title("Convergence - TSP")
        plot.xlabel('iterations')
        plot.ylabel('loss')
        plot.plot(x_axis, distances)
        plot.show()

        return route, total_distance, i, distances

    def calculate_bias(self):
        """
        Compute the bias term based on the current solution.

        Returns:
            numpy.ndarray: Bias values calculated for each city.
        """
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
        """
        Compute the total distance of the current TSP route.

        Returns:
            float: Total distance of the computed route.
        """
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
        """
        Print the computed TSP solution and its total distance.
        """
        route, total_distance = self.calculate_total_distance()

        print("Route: ", route)
        print("Total distance: ", total_distance)
