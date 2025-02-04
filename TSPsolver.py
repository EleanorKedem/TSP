import numpy as np
import random
import math
import matplotlib.pyplot as plot
import cProfile
import pstats


class TSPSolver:
    """
    A Simulated Annealing (SA) based solver for the Traveling Salesman Problem (TSP).
    """
    def __init__(self, distance_matrix, initial_temperature=100000, cooling_rate=0.99999, max_iterations=1000000):
        """
        Initialises the TSP solver with the given parameters.

        Args:
            distance_matrix: 2D NumPy array representing the distances between cities.
            initial_temperature: Initial temperature for simulated annealing.
            cooling_rate: Rate at which the temperature cools.
            max_iterations: Maximum number of iterations for optimisation.
        """

        self.num_cities = distance_matrix.shape[0]  # Number of cities in the problem
        self.distance_matrix = distance_matrix  # Distance matrix storing distances between cities
        c = 1400
        k = 0.08
        self.initial_temperature = self.num_cities*1000#15000 #* self.num_cities #c * math.exp(k * self.num_cities)  # Dynamic initial temperature
        self.cooling_rate = 1 - 1/self.initial_temperature  # Cooling schedule as a parameter of the initial temperature
        self.max_iterations = self.num_cities*50000  # Maximum allowed iterations as a parameter of the problem size

    def calculate_total_distance(self, path):
        """
         Calculate the solution based on the TSP matrix.

         Args:
             path (list): List representing the solution.

         Returns:
             float: Distance of the solution.
        """
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.distance_matrix[int(path[i])][int(path[i + 1])]
        total_distance += self.distance_matrix[int(path[-1])][int(path[0])]  # Return to the starting city
        return total_distance

    def error(self, route, optDist):
        """
        Compute the difference between the current solution's distance and the optimal solution.

        Args:
            route (list): Current solution.
            optDist (float): Optimal solution energy.

        Returns:
            float: Distance difference.
        """
        d = self.calculate_total_distance(route)
        return d - optDist

    def get_neighbour(self, path):
        """
        Generate a neighboring solution by randomly swapping cities.
        Args:
            path (list): Current solution vector.

        Returns:
            list: New solution vector with swapped cities.
        """
        i, j = random.sample(range(self.num_cities), 2)
        new_path = path[:]
        new_path[i], new_path[j] = path[j], path[i]
        return new_path

    def minimise(self, optDist):
        """
        Perform simulated annealing to minimise the TSP problem.

        Args:
            optDist (float): Optimal solution for comparison.

        Returns:
            tuple: Best solution, its energy, convergence step, and error values over iterations.
        """
        print("number of iterations %d, initial temperature %d, cooling rate %f" % (self.max_iterations, self.initial_temperature, self.cooling_rate))
        current_path = list(range(self.num_cities))  # Initial random solution
        random.shuffle(current_path)
        current_distance = self.calculate_total_distance(current_path)
        best = current_distance
        best_iter = self.max_iterations
        distances = []

        temperature = self.initial_temperature
        flag = 0
        err = self.error(current_path, optDist)
        iteration = 0
        converge_flag = False
        prev_err = err
        converge_counter = 0
        converge = self.max_iterations

        # profiler = cProfile.Profile()
        # profiler.enable()

        while iteration < self.max_iterations and converge_counter < 1000 and err > 0.01*optDist:
            new_path = self.get_neighbour(current_path)
            new_distance = self.calculate_total_distance(new_path)

            if new_distance < current_distance:
                best = new_distance
                best_iter = iteration
            # Check if the new solution is better or accept it with a probability
            if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temperature):
                current_path = new_path
                current_distance = new_distance

            temperature *= self.cooling_rate
            if temperature < 0.01 and flag == 0:
                print("temperature is zero, iteration is %d" % iteration)
                flag = 1
                print("error %d" %err)

            prev_err = err
            err = self.error(current_path, optDist)

            if err == prev_err:
                if converge_flag == False:
                    converge = iteration
                converge_flag = True
                converge_counter += 1

            else:
                converge_counter = 0
                converge_flag = False


            distances.append((current_distance-optDist)/optDist)

            # print("iteration: ", iteration)
            # print("temperature: ", temperature)
            # print("Best Path:", current_path)
            # print("Best Distance:", current_distance)

            iteration += 1

        #profiler.disable()

        # Print out the stats
        #stats = pstats.Stats(profiler)
        #stats.sort_stats('cumtime').print_stats(10)  # Sort by cumulative time and show top 10 functions

        x_axis = range(0, len(distances))
        plot.title("Convergence -TSP")
        plot.xlabel('iterations')
        plot.ylabel('loss')
        plot.plot(x_axis, distances)
        plot.show()

        print("Best result %f, in iteration %d" %(best, best_iter))

        return current_path, current_distance, converge, distances