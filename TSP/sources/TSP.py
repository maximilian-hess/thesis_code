from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

from ortools.constraint_solver.pywrapcp import SolutionCollector
import itertools

from ortools.sat.python import cp_model
from typing import DefaultDict


class TSP:
    """
    Represents a TSP instance and provides useful utils to solve the TSP with different solvers or sample states for the qusntum tree search with different strategies.
    """
    def __init__(self, adj_matrix):
        """
        Initializes a TSP
        adj_matrix: N x N symmetric matrix defining the distances between nodes
        """
        self.adj_matrix = adj_matrix
        self.n = len(adj_matrix)

    def create_model(self, threshhold = None):
        """Creates a google OR Tools model to represent a TSP in QUBO form. If threshhold is None, the TSP distance is minimized, otherwise a constraint is added to find solutions below the threshhold."""
        model = cp_model.CpModel()
        vars = {}
        tmp_vars = {}
        for i in range(self.n):
            for j in range(self.n):
                name = '{0}{1}'.format(i, j)
                var = model.NewBoolVar(name)
                vars[(i,j)] = var

        cost = 0
        for i in range(self.n):
            step_count = 0
            node_count = 0
            for j in range(self.n):
                step_count += vars[(i, j)]
                node_count += vars[(j, i)]
                for t in range(self.n):
                    tmp = model.NewBoolVar('{0}{1}{2}'.format(i, j, t))
                    tmp_vars[(i, j, t)] = tmp
                    model.AddMultiplicationEquality(tmp, vars[(i, t)], vars[(j, (t + 1) % self.n)])
                    cost += int(self.adj_matrix[i][j]) * tmp
            model.Add(step_count == 1)
            model.Add(node_count == 1)
        if not threshhold is None:
            model.Add(cost < threshhold)
        else:
            model.Minimize(cost)
        return model, vars
    
    def solutions_above_threshhold(self, threshhold):
        """Computes all TSP solutions above a threshhold with Google OR Tools. This function uses the one-hot encoding with N^2 binary variables for N nodes."""
        model, vars = self.create_model(threshhold)
        solver = cp_model.CpSolver()
        solution_collector = CustomSolutionCollector(vars, self.n)
        # Enumerate all solutions
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.max_time_in_seconds = 20000
        solver.parameters.log_search_progress = True
        #solver.log_callback = print
        # Solve to find all solutions
        #status = solver.Solve(model, solution_callback=solution_collector)
        status = solver.SearchForAllSolutions(model, solution_collector)
        print('Status with', str(self.n), 'nodes:', status)
        solutions = solution_collector.solutions

        interpreted = []
        for solution in solutions:
            s = self.interpret_solution(solution)
            if self.evaluate(s) < threshhold or True:
                interpreted.append(s)

        return interpreted
    
    def create_model_efficient(self, threshhold=None):
        """
        Creates a google OR Tools model to represent a TSP in a permutation form. This format uses N integer variables instead of binary variables 
        If threshhold is None, the TSP distance is minimized, otherwise a constraint is added to find solutions below the threshhold.
        """
        model = cp_model.CpModel()
        n = self.n
        max_value = np.max(self.adj_matrix)
        adj_list = list(self.adj_matrix)
        route = []
        for i in range(n):
            route.append(model.NewIntVar(0, n - 1, 'step' + str(i)))
        
        model.AddAllDifferent(route)

        cost = []
        cost_function = 0
        for step in range(n):
            end_points = []
            for i in range(n):
                end_points.append(model.NewIntVar(0, max_value, 'end_point' + str(i) + '-step' + str(step) ))
                model.AddElement(route[(step + 1) % n], adj_list[i], end_points[-1])
            cost.append(model.NewIntVar(0, max_value, 'cost' + str(step)))
            model.AddElement(route[step], end_points, cost[-1])
            cost_function += cost[step]
        
        if threshhold is None:
            model.Minimize(cost_function)
        else:
            model.Add(cost_function < threshhold)

        return model, route

    def solutions_above_threshhold_efficient(self, threshhold=None):
        """
        Computes all TSP solutions below the threshhold with OR Tools and the permutation form of the model. 
        """
        model, route = self.create_model_efficient(threshhold)

        solver = cp_model.CpSolver()
        solution_collector = CustomSolutionCollector2(route, self.n)
        # Enumerate all solutions
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.max_time_in_seconds = 20000

        status = solver.SearchForAllSolutions(model, solution_collector)
        print('Status with', str(self.n), 'nodes:', status)
        solutions = solution_collector.solutions
        return solutions
    
    def get_probability(self, state: list[int], weights: list, reference: list[int]):
        """
        Computes the probability of sampling a certain state after the quantum state preparation given a reference state.
        The list of weights denote the probability of sampling the node from the reference state at each TSP step.
        The remaining probability is distributed among all other remaining nodes. If the reference node is not available, one of the remaining nodes is drawn uniformly.
        This is not compatible with Lin-Kernighan inspired state preparations, use the sampling functions instead.
        """
        prob = 1
        n = self.n

        for i in range(n - 1):
            if state[i] == reference[i]:
                prob = prob * weights[i]
            elif reference[i] in state[:i]:
                prob = prob / (n - i)
            else:
                prob = prob * (1 - weights[i]) / (n - i - 1)
        return prob

    def sample_state(self, weights, reference):
        """
        Randomly samples a state according to the probabilities defined in get_probability. This function is not compatible with Lin-Kernighan inspired algorithms.
        """
        n = self.n
        result = []
        for i in range(n):
            remaining = []
            reference_index = -1
            for j in range(n):
                if not j in result:
                    remaining.append(j)
                    if reference[i] == j:
                        reference_index = len(remaining) - 1
            if i != n - 1:
                remaining_size = len(remaining)
                if reference_index == -1:
                    probs = [1 / remaining_size] * remaining_size
                else:
                    probs = [1 / (remaining_size - 1) * (1 - weights[i])] * remaining_size
                    probs[reference_index] = weights[i]
                node = np.random.choice(remaining, p=probs)
                result.append(int(node))
            else:
                result.append(remaining[0])
        return result

    def sample_state_lk(self, reference: list[int], start_index: int, chain_length: int):
        """
        Randomly samples a state where a number of consecutive nodes from the reference state are swapped out for other nodes. Nodes can be swapped out for other nodes within the chain or outside of the chain.
        The circuit defining a state preparation like this is implemented in Qiskit in qtg_circuit.ipynb.
        start_index: Defines at what point in the TSP the chain starts
        chain_length: Defines how many nodes are swapped out for others
        """
        n = self.n
        result = [-1] * n
        reference = [reference[(i + start_index) % n] for i, _ in enumerate(reference)]

        reference_indices = [-1] * n
        for index, node in enumerate(reference):
            reference_indices[node] = index

        for i in range(n):
            if result[i] != -1:
                continue
            elif i  >= chain_length:
                result[i] = reference[i]
                continue

            remaining = []
            for j in range(n):
                if not j in result and reference[i] != j:
                    remaining.append(j)
            
            choice = np.random.choice(remaining)
            node = int(choice)
            result[i] = node
            result[reference_indices[node]] = reference[i]

        result = [result[(i - start_index) % n] for i, _ in enumerate(result)]
        return result
    
    def sample_node_replacement(self, reference: list[int], start_index: int, chain_length: int):
        """
        Randomly samples a state where a number of consecutive nodes from the reference state are swapped out for other nodes. Nodes can be swapped out for other nodes within the chain or outside of the chain.
        The circuit defining a state preparation like this is implemented in Qiskit in qtg_circuit.ipynb.
        start_index: Defines at what point in the TSP the chain starts
        chain_length: Defines how many nodes are swapped out for others
        """
        n = self.n
        result = [-1] * n
        reference = [reference[(i + start_index) % n] for i, _ in enumerate(reference)]

        reference_indices = [-1] * n
        for index, node in enumerate(reference):
            reference_indices[node] = index

        for i in range(n):
            if result[i] != -1:
                continue
            elif i  >= chain_length:
                result[i] = reference[i]
                continue

            remaining = []
            for j in range(n):
                if not j in result and reference[i] != j:
                    remaining.append(j)
            
            choice = np.random.choice(remaining)
            node = int(choice)
            result[i] = node
            result[reference_indices[node]] = reference[i]

        result = [result[(i - start_index) % n] for i, _ in enumerate(result)]
        return result
    

    def sample_state_k_opt(self, reference: list[int], remove_edges:list[list[int]]):
        """
        Randomly samples a state where k edges are replaced by other edges in the TSP solution.
        reference: current solution which is modified by edge replacements
        remove_edges: list of edges to be replaced. An edge is represented by a tuple of nodes, say [1,2].
        """
        n = self.n

        cutting_indices = [reference.index(edge[0]) for edge in remove_edges]
        cutting_indices.sort()
        segments = []

        for i in range(len(cutting_indices)):
            start = cutting_indices[i] + 1
            end = cutting_indices[(i + 1) % len(cutting_indices)] + 1
            if end < start:
                end += n
            segment = [reference[j % n] for j in range(start, end)]
            segments.append(segment)

        # randomly reconnect segments
        np.random.shuffle(segments)
        # randomly choose orientation for each segment
        for i in range(len(segments)):
            if np.random.rand() < 0.5:
                segments[i] = segments[i][::-1]
        # concatenate segments to form new route
        new_route = [node for segment in segments for node in segment]
        return new_route


    def interpret_solution(self, solution):
        """Decodes a solution from a one-hot encoding"""
        decoded = []
        for i in range(self.n):
            slice = solution[i * self.n : (i + 1) * self.n]
            for j in range(self.n):
                if slice[j] == 1:
                    decoded.append(j)
                    break
        return decoded

    def evaluate(self, solution):
        """Calculates the total cycle distance for a TSP solution"""
        distance = 0
        for i in range(self.n):
            distance += self.adj_matrix[solution[i]][solution[(i + 1) % self.n]]
        return distance
    
    def bitstring_to_integer_sol(self, bitstring):
        sol = [0]*self.n
        for i in range(self.n):
            for j in range(self.n):
                if bitstring[i*self.n + j] == 1:
                    sol[j] = i
        return sol
    
    def integer_sol_to_bitstring(self, sol):
        bitstring = [0]*(self.n**2)
        for i,s in enumerate(sol):
            bitstring[i*self.n + s] = 1
        return bitstring

    
    def evaluate_bitstring(self, solution):
        """Calculates the total cycle distance for a TSP solution"""
        int_sol = self.bitstring_to_integer_sol(solution)
        return self.evaluate(int_sol)    
    
    def solve_exact(self):
        """Searches for an exact solution using OR-Tools with a one-hot encoding of the TSP"""
        model, vars = self.create_model_efficient()
        solver = cp_model.CpSolver()
        solver.Solve(model)
        solution = [0] * self.n * self.n
        for key in vars:
            solution[self.n * key[1] + key[0]] = solver.Value(vars[key])
        decoded = self.interpret_solution(solution)
        return decoded, solver.best_objective_bound
    
    def solve_exact_efficient(self):
        """Searches for an exact solution using OR-Tools with a node-based encoding of the TSP"""
        model, vars = self.create_model_efficient()
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        print('Solver status', solver.StatusName(status))
        solution = []
        for index, var in enumerate(vars):
            solution.append(solver.Value(var))
        return solution, solver.ObjectiveValue()
        
    def solve_tsp_or(self):
        """Searches for a TSP solution with OR Routing. Code from https://developers.google.com/optimization/routing/tsp"""
        manager = pywrapcp.RoutingIndexManager(len(self.adj_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.adj_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        solution = routing.SolveWithParameters(search_parameters=search_parameters)
        routes = get_routes(solution, routing, manager)
        return routes[0], self.evaluate(routes[0])
    
    def solve_greedy(self):
        """Uses a greedy heuristic to solve the TSP"""
        sum = 0
        counter = 0
        j = 0
        i = 0
        INT_MAX = 2147483647
        min = INT_MAX
        visitedRouteList = DefaultDict(int)
        tsp = self.adj_matrix
    
        visitedRouteList[0] = 1
        route = [0] * len(tsp)
        result = [0] * (len(tsp) + 1)

        while i < len(tsp) and j < len(tsp[i]):

            if counter >= len(tsp[i]) - 1:
                break

            if j != i and (visitedRouteList[j] == 0):
                if tsp[i][j] < min:
                    min = tsp[i][j]
                    route[counter] = j + 1
    
            j += 1

            if j == len(tsp[i]):
                visitedRouteList[route[counter] - 1] = 1
                j = 0
                i = route[counter] - 1
                result[counter + 1] = route[counter] - 1
                counter += 1
                sum += min
                min = INT_MAX

        i = route[counter - 1] - 1

        sum += tsp[i][0]

        return result[:-1], sum

def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes
    
def remove_duplicates(k):
    k.sort()
    return list(k for k,_ in itertools.groupby(k))
    
class CustomSolutionCollector(cp_model.CpSolverSolutionCallback):
    """Collect solutions above a certain threshold. Specific to the one-hot representation of the TSP."""

    def __init__(self, variables, n):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.n = n
        self.solutions = []

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        node = [0] * self.n * self.n
        for key in self.__variables:
            node[self.n * key[1] + key[0]] = self.Value(self.__variables[key])
        self.solutions.append(node)


    @property
    def solution_count(self) -> int:
        return self.__solution_count
    
class CustomSolutionCollector2(cp_model.CpSolverSolutionCallback):
    """Collect solutions above a certain threshold. Specific to the node-level representation of the TSP"""

    def __init__(self, variables, n):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.n = n
        self.solutions = []

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        node = []
        for step in self.__variables:
            node.append(self.Value(step))
        self.solutions.append(node)


    @property
    def solution_count(self) -> int:
        return self.__solution_count

