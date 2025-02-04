import numpy as np
import matplotlib.pyplot as plot
import cProfile
import pstats

class QMLsolver:
    """
    Quantum-inspired Machine Learning (QML) solver for the Traveling Salesman Problem (TSP).

    Uses simulated annealing with tunable parameters for optimization.
    """

    def __init__(self, distance_matrix, initial_temperature=100000, cooling_rate=0.99999, max_iterations=1000000):
        """
         Initialize the QMLsolver with problem parameters.

         Args:
             distance_matrix (np.ndarray): TSP distance matrix.
             initial_temperature (float, optional): Initial temperature for annealing. Default is 100000.
             cooling_rate (float, optional): Cooling rate for annealing. Default is 0.99999.
             max_iterations (int, optional): Maximum number of iterations. Default is 1000000.
        """
        self.num_cities = distance_matrix.shape[0]
        self.distance_matrix = distance_matrix
        self.initial_temperature = self.num_cities*100000  # Temperature scaling based on problem size
        self.cooling_rate = 1 - 10/self.initial_temperature  # Adaptive cooling rate as a parameter of the initial temperature
        self.max_iterations = self.num_cities*500000  # Adaptive iteration limit as a parameter of the problem size

#input - list of cities as the route
#output - the length of the route
    def total_dist(self, route):
        """
         Calculate the solution based on the TSP matrix.

         Args:
             route (list): List representing the solution.

         Returns:
             float: Distance of the solution.
        """

        d = 0.0  # total distance between cities

        for i in range(self.num_cities):
           d += self.distance_matrix[int(route[i])][int(route[i+1])]
        return d

    def error(self, route, optDist):
        """
        Compute the difference between the current solution's distance and the optimal solution.

        Args:
            route (list): Current solution.
            optDist (float): Optimal solution energy.

        Returns:
            float: Distance difference.
        """

        d = self.total_dist(route)
        error = d - optDist
        return error

    def adjacent(self, route, n_swaps, rnd):
        """
        Generate a neighboring solution by randomly swapping cities.
        Args:
            route (list): Current solution vector.
            n_swaps (int): Number of cities to swap.
            rnd (np.random.RandomState): Random generator instance.

        Returns:
            list: New solution vector with swapped cities.
        """
        n = len(route)
        result = np.copy(route)
        for ns in range(n_swaps):
            #since we start and return to the first city we need to deduct 2 from the length of the route
            i = rnd.randint(1, (n-2))
            j = rnd.randint(1, (n-2))
            tmp = result[i]
            result[i] = result[j]
            result[j] = tmp
        return result

    def my_kendall_tau_dist(self, p1, p2):
        """
        Compute the Kendall tau distance between two sequences.

        Args:
            p1 (list): First sequence.
            p2 (list): Second sequence.

        Returns:
            tuple: Raw distance (number of pair misorderings) and normalised distance.
        """
        # p1, p2 are 0-based lists or np.arrays
        n = len(p1)
        index_of = [None] * n  # lookup into p2
        for i in range(n):
            v = p2[i]
            index_of[v] = i

        d = 0  # raw distance = number pair misorderings
        for i in range(n):
            for j in range(i+1, n):
                if index_of[p1[i]] > index_of[p1[j]]:
                    d += 1
        normer = n * (n - 1) / 2.0
        nd = d / normer  # normalized distance
        return (d, nd)

    def minimise(self, optDist, pctTunnel=0.25):
        """
        Perform simulated annealing to minimise the TSP problem.

        Args:
            optDist (float): Optimal solution for comparison.
            pctTunnel (float, optional): Probability of tunneling move. Default is 0.15.

        Returns:
            tuple: Best solution, its energy, convergence step, and error values over iterations.
        """
        print("number of iterations %d, initial temperature %d, cooling rate %f" % (
        self.max_iterations, self.initial_temperature, self.cooling_rate))

        rnd = np.random.RandomState(30)
        currTemp = self.initial_temperature
        soln = np.arange(1, self.num_cities, dtype=np.int64)
        rnd.shuffle(soln)
        soln = np.insert(soln, 0, 0)
        soln = np.append(soln, 0)
        print("Initial guess: ")
        print(soln)

        err = self.error(soln, optDist)
        iteration = 0
        interval = (int)(self.max_iterations / 10)

        bestSoln = np.copy(soln)
        bestErr = err
        distances = []

        currConverge = 0
        converge = 0
        converge_threshold = 1e-3

        #profiler = cProfile.Profile()
        #profiler.enable()

        while err > converge_threshold and iteration < self.max_iterations:
            # pct left determines n_swaps determines distance
            pct_iters_left = (self.max_iterations - iteration) / (self.max_iterations * 1.0)
            p = rnd.random()  # [0.0, 1.0]
            if p < pctTunnel:            # tunnel
                numSwaps = (int)(pct_iters_left * self.num_cities)
                if numSwaps < 1:
                     numSwaps = 1
            else: # no tunneling
                numSwaps = 1

            adjRoute = self.adjacent(soln, numSwaps, rnd)
            adjErr = self.error(adjRoute, optDist)
            prevErr = err

            if adjErr < bestErr:
                bestSoln = np.copy(adjRoute)
                bestErr = adjErr

            if adjErr < err:  # better route so accept
                soln = adjRoute
                err = adjErr

            else: # adjacent is worse
                accept_p = np.exp((err - adjErr) / currTemp)
                p = rnd.random()
                if p < accept_p:  # accept anyway
                    soln = adjRoute
                    err = adjErr
                # else don't accept worse route

            if abs(prevErr - err) < converge_threshold:
                if converge < interval: #incrementing converge for stats only for interval
                    converge += 1
            else:
                converge = 0
                converge_start = iteration

            if iteration % interval == 0:
                (dist, nd) = self.my_kendall_tau_dist(soln, adjRoute)
                print("iteration = %6d | " % iteration, end="")
                print("dist curr to candidate = %8.4f | " % nd, end="")
                print("curr_temp = %12.4f | " % currTemp, end="")
                print("error = %6.1f " % (bestErr/optDist))
                print("converge %d " % converge_start)
                if currConverge == converge_start and iteration > 0:
                    print("I'm here!")
                    break
                else:
                    currConverge = converge_start

            if currTemp < 0.00001:
                currTemp = 0.00001
            else:
                currTemp *= self.cooling_rate

            distances.append(err/optDist)
            iteration += 1

        #profiler.disable()

        # Print out the stats
        #stats = pstats.Stats(profiler)
        #stats.sort_stats('cumtime').print_stats(10)  # Sort by cumulative time and show top 10 functions

        print("error %f converge %d" %(bestErr, converge_start))
        energy = self.total_dist(bestSoln)
        print(bestSoln)
        x_axis = range(0, len(distances))
        plot.title("Convergence - TSP")
        plot.xlabel('iterations')
        plot.ylabel('loss')
        plot.plot(x_axis, distances)
        plot.show()

        return bestSoln, energy, converge_start, distances

