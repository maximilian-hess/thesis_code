import numpy as np
from collections import defaultdict


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
    def evaluate(self, solution):
        """Calculates the total cycle distance for a TSP solution"""
        distance = 0
        for i in range(self.n):
            distance += self.adj_matrix[solution[i]][solution[(i + 1) % self.n]]
        return distance
    
    def solve_greedy(self):
        """Uses a greedy heuristic to solve the TSP"""
        sum = 0
        counter = 0
        j = 0
        i = 0
        INT_MAX = 2147483647
        min = INT_MAX
        visitedRouteList = defaultdict(int)
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

