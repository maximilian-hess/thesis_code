import numpy as np

class MultipleKnapsackProblem:
    """Multiple Knapsack Problem implementation"""
    
    def __init__(self, article_reward, article_weight, knapsack_capacity):
        """
        Parameters:
        - article_reward: list of lists, reward[k][i] for item i in knapsack k
                         Shape: (num_knapsacks, num_items)
        - article_weight: list, weight of each item, Shape: (num_items,)
        - knapsack_capacity: list, capacity of each knapsack, Shape: (num_knapsacks,)
        """
        self.article_reward = np.array(article_reward)  # Shape: (m, n)
        self.article_weight = np.array(article_weight)   # Shape: (n,)
        self.knapsack_capacity = np.array(knapsack_capacity)  # Shape: (m,)
        
        self.num_knapsacks = len(knapsack_capacity)
        self.num_items = len(article_weight)

        self.penalty_capacity = 2*max(self.article_reward.flatten())  # Penalty for item in multiple knapsacks
        self.penalty_assignment = 2*max(self.article_reward.flatten())    # Penalty for capacity constraint
    
    def greedy_solution(self):
        """"
        Outputs a greedy solution for the item order provided
        """
        bitstring = [0 for _ in range(self.num_items*self.num_knapsacks)]
        value = 0
        remaining_cap = [c for c in self.knapsack_capacity]
        item_assigned = [False for _ in range(self.num_items)]
        for item in range(self.num_items):
            for knap in range(self.num_knapsacks):
                if remaining_cap[knap] >= self.article_weight[item] and item_assigned[item] == False:
                    bitstring[self.num_items*knap + item] = 1
                    item_assigned[item] = True
                    remaining_cap[knap] -= self.article_weight[item]
                    value += self.article_reward[knap][item]
                    break # item is already assigned, do not need to iterate through all knapsacks
        return bitstring, value

    def evaluate_bitstring(self, bitstring, penalty_assignment=None, penalty_capacity=None, s=None, unbalanced_penalization=False):
        # Use instance penalties if not provided
        if penalty_assignment is None:
            penalty_assignment = self.penalty_assignment
        if penalty_capacity is None:
            penalty_capacity = self.penalty_capacity
        
        n = self.num_items
        m = self.num_knapsacks
        
        # Decode bitstring into assignment matrix
        assignment = np.zeros((n, m), dtype=int)
        
        for i in range(n):
            for j in range(m):
                bit_idx = i * m + j
                if bit_idx < len(bitstring) and int(bitstring[bit_idx]) == 1:
                    assignment[i, j] = 1
        
        # Initialize objective value
        objective = 0.0
        
        # ============================================================
        # Part 1: Reward (negative because we minimize in Hamiltonian)
        # ============================================================

        for i in range(n):
            for j in range(m):
                if assignment[i, j] == 1:
                    objective -= self.article_reward[j][i]
        
        # ============================================================
        # Part 2: "At most one knapsack" constraint penalty
        # H_assignment = λ₁ Σᵢ Σⱼ<ₖ x_{i,j} x_{i,k}
        # ============================================================
        assignment_penalty = 0.0
        for i in range(n):
            num_assignments = assignment[i, :].sum()
            if num_assignments > 1:
                # Count pairs: C(num_assignments, 2) = num_assignments * (num_assignments - 1) / 2
                num_pairs = num_assignments * (num_assignments - 1) // 2
                assignment_penalty += num_pairs
        
        objective += penalty_assignment * assignment_penalty
        
        # ============================================================
        # Part 3: Capacity constraint penalty
        # H_capacity = λ₂ Σⱼ (Σᵢ wᵢ x_{i,j} - Cⱼ)²
        # ============================================================
        capacity_penalty = 0.0
        for j in range(m):
            total_weight = 0.0
            for i in range(n):
                if assignment[i, j] == 1:
                    total_weight += self.article_weight[i]
            
            # Penalty is the squared violation
            
            # continuous slack variable variant
            if s is not None:
                violation = (total_weight + s[j] - self.knapsack_capacity[j])**2
                capacity_penalty += violation
            # no slack variable variant
            elif not unbalanced_penalization:
                violation = max([total_weight - self.knapsack_capacity[j],0])**2
                capacity_penalty += violation
            elif unbalanced_penalization:
                violation = 0.5*(total_weight - self.knapsack_capacity[j])**2 + (total_weight - self.knapsack_capacity[j])
                capacity_penalty += violation
        
        objective += penalty_capacity * capacity_penalty
        
        return objective
    
    def simple_evaluate(self, bitstring):
        """Evaluate a bitstring and return its value or 'infeasible' status"""
        remaining_caps = [c for c in self.knapsack_capacity]
        items_assigned = [False for _ in range(self.num_items)]
        value = 0
        for ind, b in enumerate(bitstring):
            i = ind % self.num_items  # item number
            k = int(np.floor(ind / self.num_items))  # knapsack number
            if b == 1:
                if items_assigned[i]:
                    print(f"item assigned twice at {ind} by item {i} and knapsack {k}")
                    return "infeasible"
                remaining_caps[k] -= self.article_weight[i]
                if remaining_caps[k] < 0:
                    print(f"capacity surpassed at {ind} by item {i} and knapsack {k}")
                    return "infeasible"
                items_assigned[i] = True
                value += self.article_reward[k][i]
        return value
    
    def get_num_qubits(self):
        """Return number of qubits needed: n × m"""
        return self.num_items * self.num_knapsacks
