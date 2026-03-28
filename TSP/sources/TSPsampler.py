from sources.TSP import TSP
import numpy as np


class TSP_sampler:
    def __init__(self, tsp: TSP):
        self.tsp = tsp
        self.n = tsp.n

    def sample_lk(self, initial_reference, l_max: int, T_max: int):
        """"
        
        initial_reference: initial tour to adjust
        l_max: maximum chain length
        T_max: maximum number of samples
        """
        t_total = 0
        ref = initial_reference
        val = self.tsp.evaluate(ref)
        improvement_found = True
        while improvement_found:
            improvement_found = False
            for l in range(1,l_max+1):
                # iterate through all starting points
                for i in range(self.n):
                    # use sample_lk function from TSP class to sample LK states with chain length l and starting point i
                    # limit number of iterations with one setting of l and i to T_max / (l * n)
                    internal_limit = T_max // (l * self.n * int(np.log(self.n)))
                    for _ in range(internal_limit):
                        sol_candidate = self.tsp.sample_state_lk(reference=ref, start_index=i, chain_length=l)
                        t_total += 1
                        if self.tsp.evaluate(sol_candidate) < val:
                            # update reference solution
                            ref = sol_candidate
                            val = self.tsp.evaluate(ref)
                            print("Improved solution found with value:", val, "total samples:", t_total)
                            improvement_found = True
                            break
                    if improvement_found:
                        break
                if improvement_found:
                    break
        return ref, val, t_total

    def sample_node_replacement(self, initial_reference, k: int, T_max: int):
        """"
        initial_reference: initial tour to adjust
        k: number of nodes to swap
        T_max: maximum number of samples
        """
        t_total = 0
        ref = initial_reference
        val = self.tsp.evaluate(ref)

        while t_total < T_max:
            # randomly select starting point
            i = np.random.randint(0, self.n)
            sol_candidate = self.tsp.sample_state_lk(reference=ref, start_index=i, chain_length=k)
            t_total += 1
            if self.tsp.evaluate(sol_candidate) < val:
                # update reference solution
                ref = sol_candidate
                val = self.tsp.evaluate(ref)
                print("Improved solution found with value:", val, "total samples:", t_total)
        return ref, val, t_total

    
    def sample_k_opt(self, initial_reference, k: int, T_max: int):
        """"
        initial_reference: initial tour to adjust
        k: number of nodes to swap
        T_max: maximum number of samples
        """
        t_total = 0
        ref = initial_reference
        val = self.tsp.evaluate(ref)
        print("Initial solution value:", val)
        while t_total < T_max:
            for _ in range(T_max):
                removal_indices = np.random.choice(len(ref), size=k, replace=False)
                removed_edges = [[ref[idx], ref[(idx + 1) % len(ref)]] for idx in removal_indices]
                sol_candidate = self.tsp.sample_state_k_opt(reference=ref, remove_edges=removed_edges)
                t_total += 1
                if self.tsp.evaluate(sol_candidate) < val:
                    # update reference solution
                    ref = sol_candidate
                    val = self.tsp.evaluate(ref)
                    print("Improved solution found with value:", val, "total samples:", t_total)
                    break
        return ref, val, t_total