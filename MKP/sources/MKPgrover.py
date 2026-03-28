import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from scipy.optimize import minimize
try:
    from .MKP import MultipleKnapsackProblem as MultiKnapsack
except ImportError:
    from MKP import MultipleKnapsackProblem as MultiKnapsack


class MKPGrover:
    def __init__(self, mkp:MultiKnapsack) -> None:
        self.mkp = mkp
        self.greedy_sol, self.value = self.mkp.greedy_solution()

    def iterative_solver_gas(self, initial_threshold: int, bitstrings: list, L=2, branching_weights=None):
        """"
        Grover adaptive search for a multiple knapsack problem.
        States are prepared with different biasing techniquess
        initial_threshold: initial threshold (e.g. from a heuristic)
        original Grover adaptive search https://arxiv.org/pdf/1912.04088
        bitstrings: list of bitstrings above the initial threshold (precomputed to reduce computational cost)
        L: "exponential search parameter" from GAS/original Gilles-Brassard paper
        """
        # number of variables in multi knapsack problem
        N = self.mkp.num_items * self.mkp.num_knapsacks
        iteration_limit = N
        improvement_possible = True
        current_threshold = initial_threshold
        current_sol = None
        # parameter from GAS paper/Gilles-Brassard "exponential search"
        L = 5/4
        # outer loop: we stop when we can't find a better solution
        total_iterations = 0
        # count number of threshold updates
        threshold_updates = 0

        while improvement_possible:
            # count iterations within the outer loop
            iterations = 0
            k = 1
            # amount of times we have repeated our randomized strategy
            trials = 0         
            # inner loop: every single Grover search is run until either a solution is found or the number of iteration is maxed out


            # read solutions above current threshold from list of bitstrings
            sols = [bitstring for bitstring in bitstrings if self.mkp.simple_evaluate(bitstring) > current_threshold]
            no_sols = len(sols)
            
            while iterations < iteration_limit:
                improvement_possible = True
                # choose number of Grover iterations randomly from an increasing range
                grover_iterations = np.random.randint(0, np.ceil(L**k))
                total_iterations += grover_iterations
                iterations += grover_iterations

                # set up probabilities for good states after state prep
                pre_grover_probs = [self.bitstring_prob(bitstring) for bitstring in sols]
                P = sum(pre_grover_probs)
                # measurement probability for every single good solution after Grover search
                post_grover_probs = [prob_after_grover(p, P, grover_iterations) for p in pre_grover_probs]

                probs = post_grover_probs + [1-sum(post_grover_probs)]
                outcomes = np.array(sols + ["no improvement"], dtype=object)

                sampled_bitstring = np.random.choice(outcomes, p=probs)
                if sampled_bitstring != "no improvement":
                    print("improvement found")
                    current_threshold = self.mkp.simple_evaluate(sampled_bitstring)
                    current_sol = sampled_bitstring
                    threshold_updates += 1
                    # update branching probabilities after threshold update
                    if branching_weights[0]>0:
                        self.set_branching_probs(branching_weights=branching_weights, current_sol=current_sol)
                    break
                else:
                    print("no improvement found")
                    improvement_possible = False
                    iterations += grover_iterations
                    k += 1
                    trials += 1
        iterations_wo_improvement = iterations
        return current_sol, current_threshold, total_iterations, threshold_updates

    def set_branching_probs(self, branching_weights, current_sol):
        """"
        Set branching probabilities according to the provided branching_weights
        branching_weight must have length 3 and sum up to 1
        We will use three criteria:
        1. proximity to reference solution
        2. favor assignments of 1
        3. tree balancing
        """
        if current_sol is None:
            current_sol = self.greedy_sol
        p_bias = 0.75
        nbh_probs = []
        obj_probs = []
        tb_probs = []
        n = self.mkp.num_items
        m = self.mkp.num_knapsacks
        for ind,b in enumerate(current_sol):
            if b==0:
                nbh_probs.append(p_bias)
            else:
                nbh_probs.append(1-p_bias)
            obj_probs.append(1-p_bias)
            item = np.floor(ind/m)
            tb_probs.append((((n-1) - item)/(n-1))*p_bias + ((item)/(n-1))*(1-p_bias))

        self.branching_probs = []
        for ind in range(n*m):
            prob = branching_weights[0]*nbh_probs[ind] + branching_weights[1]*obj_probs[ind] + branching_weights[2]*tb_probs[ind]
            self.branching_probs.append(prob)





    
    def bitstring_prob(self, bitstring):
        """"
        Given a MKP bitstring [x_1,...,x_{m*n}] return its initial measurement probability
        """
        prob = 1
        remaining_cap = [c for c in self.mkp.knapsack_capacity]
        items_assigned = [False for _ in range(self.mkp.num_items)]
        for ind,b in enumerate(bitstring):
            i = ind % self.mkp.num_items # item number
            k = int(np.floor(ind/self.mkp.num_items)) # knapsack number
            if remaining_cap[k] >= self.mkp.article_weight[i] and items_assigned[i]==False: # if branching condition is fulfilled
                if b == 0:
                    prob = prob*self.branching_probs[ind]
                else:
                    prob = prob*(1-self.branching_probs[ind])
                    items_assigned[i] = True
                    remaining_cap[k] -= self.mkp.article_weight[i]
        return prob







            


def prob_after_grover(p_0: float, P: float, n: int):
    """
    Calculate probability of measuring a good state after Grover iterations.
    
    Args:
        p_0: initial measurement probability for a specific state
        P: initial measurement probability for all good states
        n: number of Grover iterations
    """
    return p_0*(np.sin((2*n+1)*np.arcsin(np.sqrt(P)))**2)/P