from sources.TSP import TSP
import numpy as np


class TSP_sampler:
    def __init__(self, tsp: TSP):
        self.tsp = tsp
        self.n = tsp.n

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