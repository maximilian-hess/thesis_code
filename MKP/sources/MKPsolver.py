import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from scipy.optimize import minimize
try:
    from .MKP import MultipleKnapsackProblem
except ImportError:
    from MKP import MultipleKnapsackProblem

class MKPQAOASolver:
    """QAOA solver for Multiple Knapsack Problem"""
    
    def __init__(self, mkp_instance:MultipleKnapsackProblem, p=1, use_slack=False, cont_slack=False, unbalanced_penalization=False):
        """
        Parameters:
        - mkp_instance: MultipleKnapsackProblem instance
        - p: number of QAOA layers
        - use_slack: if True, use slack variables for capacity constraints
        """
        self.mkp = mkp_instance
        self.p = p
        self.use_slack = use_slack
        self.num_qubits_items = mkp_instance.get_num_qubits()
        self.P_a = 2*max(self.mkp.article_reward.flatten())  # Penalty for item in multiple knapsacks
        self.P_c = 2*max(self.mkp.article_reward.flatten())    # Penalty for capacity constraint
        # flag if we use the continuous slack method
        self.cont_slack = cont_slack
        self.unbalanced_penalization = unbalanced_penalization
        if unbalanced_penalization and cont_slack:
            print("Can only use one strategy out of unbalanced penalization and continuous slack!")
            self.unbalanced_penalization = False
        
        if use_slack:
            # Calculate slack qubits needed for each knapsack
            self.num_slack_per_knapsack = [
                int(np.ceil(np.log2(C + 1))) 
                for C in mkp_instance.knapsack_capacity
            ]
            self.total_slack_qubits = sum(self.num_slack_per_knapsack)
            self.num_qubits = self.num_qubits_items + self.total_slack_qubits
        else:
            self.num_slack_per_knapsack = []
            self.total_slack_qubits = 0
            self.num_qubits = self.num_qubits_items
        
        self.simulator = Sampler()
        
    def create_cost_hamiltonian_no_slack(self):
        """
        Create cost Hamiltonian WITHOUT slack variables.
        Uses penalty method for capacity constraints.
        
        Returns: linear_coeffs, quadratic_coeffs
        """
        n = self.mkp.num_items
        m = self.mkp.num_knapsacks
        num_qubits = n * m
        
        linear_coeffs = np.zeros(num_qubits)
        quadratic_coeffs = {}
        
        penalty_assignment = 200  # Penalty for item in multiple knapsacks
        penalty_capacity = 150    # Penalty for capacity violation
        
        # Objective: maximize Σᵢ Σⱼ r_{i,j} x_{i,j}
        # We minimize, so negate the rewards
        for i in range(n):
            for j in range(m):
                qubit_idx = i * m + j
                linear_coeffs[qubit_idx] = -self.mkp.article_reward[j, i]
        
            # Constraint 1: Each item in at most one knapsack
            # Penalty: λ₁ Σⱼ Σₖ>ⱼ x_{i,j}x_{i,k} -> penalty whenver item i appears in two knapsacks
            # Quadratic terms: 2 x_{i,j} x_{i,k} for j < k
            for j in range(m):
                for k in range(j+1, m):
                    qubit_j = i * m + j
                    qubit_k = i * m + k
                    if (qubit_j, qubit_k) in quadratic_coeffs:
                        quadratic_coeffs[(qubit_j, qubit_k)] += penalty_assignment
                    else:
                        quadratic_coeffs[(qubit_j, qubit_k)] = penalty_assignment
        
        # Constraint 2: Capacity for each knapsack
        # Penalty: λ₂ Σⱼ (Σᵢ w_i x_{i,j} - C_j)²
        for j in range(m):
            C_j = self.mkp.knapsack_capacity[j]
            
            # Linear terms: w_i² x_{i,j} - 2 C_j w_i x_{i,j}
            for i in range(n):
                qubit_idx = i * m + j
                w_i = self.mkp.article_weight[i]
                linear_coeffs[qubit_idx] += penalty_capacity * (w_i**2 - 2 * C_j * w_i)
                # additional term if we use unbalanced penalization
                if self.unbalanced_penalization:
                    linear_coeffs[qubit_idx] += penalty_capacity*w_i

            
            # Quadratic terms: 2 w_i w_k x_{i,j} x_{k,j} for i < k
            for i in range(n):
                for k in range(i+1, n):
                    qubit_i = i * m + j
                    qubit_k = k * m + j
                    coeff = 2 * penalty_capacity * self.mkp.article_weight[i] * self.mkp.article_weight[k]
                    if (qubit_i, qubit_k) in quadratic_coeffs:
                        quadratic_coeffs[(qubit_i, qubit_k)] += coeff
                    else:
                        quadratic_coeffs[(qubit_i, qubit_k)] = coeff
            
            # Constant C_j² omitted (doesn't affect optimization)
        
        return linear_coeffs, quadratic_coeffs
    
    def create_cost_hamiltonian_with_slack(self):
        """
        Create cost Hamiltonian WITH slack variables.
        Encodes capacity constraints exactly using slack qubits.
        
        Constraint: Σᵢ w_i x_{i,j} + Σₖ 2^k s_{j,k} = C_j
        
        Returns: linear_coeffs, quadratic_coeffs
        """
        n = self.mkp.num_items
        m = self.mkp.num_knapsacks
        
        linear_coeffs = np.zeros(self.num_qubits)
        quadratic_coeffs = {}
        
        penalty_assignment = 2*max(self.mkp.article_reward.flatten())  # Penalty for item in multiple knapsacks
        penalty_capacity = 2*max(self.mkp.article_reward.flatten())    # Penalty for capacity constraint
        
        # Objective: maximize Σᵢ Σⱼ r_{i,j} x_{i,j}
        for i in range(n):
            for j in range(m):
                qubit_idx = i * m + j
                linear_coeffs[qubit_idx] = -self.mkp.article_reward[j, i]
        
        # Constraint 1: Each item in at most one knapsack (same as before)
        for i in range(n):
            for j in range(m):
                for k in range(j+1, m):
                    qubit_j = i * m + j
                    qubit_k = i * m + k
                    if (qubit_j, qubit_k) in quadratic_coeffs:
                        quadratic_coeffs[(qubit_j, qubit_k)] += penalty_assignment
                    else:
                        quadratic_coeffs[(qubit_j, qubit_k)] = penalty_assignment
        
        # Constraint 2: Capacity with slack variables
        # For each knapsack j: (Σᵢ w_i x_{i,j} + Σₖ 2^k s_{j,k} - C_j)²
        
        slack_offset = n * m  # Slack qubits start after item qubits
        
        for j in range(m):
            C_j = self.mkp.knapsack_capacity[j]
            num_slack_j = self.num_slack_per_knapsack[j]
            
            # Get slack qubit indices for knapsack j
            slack_start = slack_offset + sum(self.num_slack_per_knapsack[:j])
            slack_qubits_j = list(range(slack_start, slack_start + num_slack_j))
            
            # Item-item interactions: 2*w_i w_k x_{i,j} x_{k,j} for all k<i
            for i in range(n):
                for k in range(i+1, n):
                    qubit_i = i * m + j
                    qubit_k = k * m + j
                    coeff = 2 * penalty_capacity * self.mkp.article_weight[i] * self.mkp.article_weight[k]
                    if (qubit_i, qubit_k) in quadratic_coeffs:
                        quadratic_coeffs[(qubit_i, qubit_k)] += coeff
                    else:
                        quadratic_coeffs[(qubit_i, qubit_k)] = coeff
            
            # Item-slack interactions: 2 w_i 2^k x_{i,j} s_{j,k}
            for i in range(n):
                qubit_i = i * m + j
                for idx, slack_qubit in enumerate(slack_qubits_j):
                    coeff = 2 * penalty_capacity * self.mkp.article_weight[i] * (2**idx)
                    key = (qubit_i, slack_qubit) if qubit_i < slack_qubit else (slack_qubit, qubit_i)
                    if key in quadratic_coeffs:
                        quadratic_coeffs[key] += coeff
                    else:
                        quadratic_coeffs[key] = coeff
            
            # Slack-slack interactions: 2 * 2^p 2^q s_{j,p} s_{j,q}
            for p in range(num_slack_j):
                for q in range(p+1, num_slack_j):
                    slack_p = slack_qubits_j[p]
                    slack_q = slack_qubits_j[q]
                    coeff = 2 * penalty_capacity * (2**p) * (2**q)
                    if (slack_p, slack_q) in quadratic_coeffs:
                        quadratic_coeffs[(slack_p, slack_q)] += coeff
                    else:
                        quadratic_coeffs[(slack_p, slack_q)] = coeff
            
            # Linear terms: w_i² x_{i,j} - 2 C_j w_i x_{i,j}
            for i in range(n):
                qubit_i = i * m + j
                w_i = self.mkp.article_weight[i]
                linear_coeffs[qubit_i] += penalty_capacity * (w_i**2 - 2 * C_j * w_i)
            
            # Linear terms for slack: (2^k)² s_{j,k} - 2 C_j 2^k s_{j,k}
            for idx, slack_qubit in enumerate(slack_qubits_j):
                power = 2**idx
                linear_coeffs[slack_qubit] += penalty_capacity * (power**2 - 2 * C_j * power)
            
            # Constant C_j² omitted
        
        return linear_coeffs, quadratic_coeffs
    
    def create_qaoa_circuit(self, gamma, beta):
        """Create QAOA circuit with given parameters"""
        qc = QuantumCircuit(self.num_qubits)
        
        # Initial state: uniform superposition
        qc.h(range(self.num_qubits))
        
        # Get Hamiltonian coefficients
        if self.use_slack:
            linear_coeffs, quadratic_coeffs = self.create_cost_hamiltonian_with_slack()
        else:
            linear_coeffs, quadratic_coeffs = self.create_cost_hamiltonian_no_slack()
        
        for layer in range(self.p):
            # Cost Hamiltonian (problem Hamiltonian)
            # Apply RZ gates for linear terms
            for i in range(self.num_qubits):
                if linear_coeffs[i] != 0:
                    qc.rz(2 * gamma[layer] * linear_coeffs[i], i)
            
            # Apply RZZ gates for quadratic terms
            for (i, j), coeff in quadratic_coeffs.items():
                if coeff != 0:
                    qc.cx(i, j)
                    qc.rz(2 * gamma[layer] * coeff, j)
                    qc.cx(i, j)
            
            # Mixer Hamiltonian (only on item qubits, not slack)
            for i in range(self.num_qubits_items):
                qc.rx(2 * beta[layer], i)
            
            # For slack qubits, also apply mixer (or could use different mixer)
            if self.use_slack:
                for i in range(self.num_qubits_items, self.num_qubits):
                    qc.rx(2 * beta[layer], i)
        
        qc.measure_all()
        return qc

    def compute_expectation(self, gamma, beta, shots=1024, s=None):
        """Compute expectation value for given parameters"""
        qc = self.create_qaoa_circuit(gamma, beta)
        
        # Run simulation
        job = self.simulator.run(qc, shots=shots)
        result = job.result()
        probs = result.quasi_dists[0].binary_probabilities()
        counts = {b: shots*probs[b] for b in probs}
        
        # Calculate expectation value
        expectation = 0
        for bitstring, count in counts.items():
            # Reverse bitstring (qiskit convention)
            bitstring_reversed = bitstring[::-1]
            
            # Extract only item qubits for evaluation
            item_bitstring = bitstring_reversed[:self.num_qubits_items]
            if s is not None:
                value = self.mkp.evaluate_bitstring(item_bitstring, penalty_assignment=self.P_a, penalty_capacity=self.P_c, s=s)
            elif self.unbalanced_penalization:
                value = self.mkp.evaluate_bitstring(item_bitstring, penalty_assignment=self.P_a, penalty_capacity=self.P_c, unbalanced_penalization=True)
            else:
                value = self.mkp.evaluate_bitstring(item_bitstring, penalty_assignment=self.P_a, penalty_capacity=self.P_c)
            expectation += value * count / shots
        
        return expectation

    def solve(self, shots=1024, maxiter=100):
        """
        Solve MKP using QAOA
        
        Returns:
        - best_bitstring: best solution found (item qubits only)
        - best_value: objective value
        - optimal_params: optimized gamma and beta parameters
        - counts: measurement counts from final circuit
        """
        # Initial parameters
        initial_gamma = np.random.uniform(0, 2*np.pi, self.p)
        initial_beta = np.random.uniform(0, np.pi, self.p)
        if not self.cont_slack:
            initial_params = np.concatenate([initial_gamma, initial_beta])
            print("Initial energy: ", self.compute_expectation(initial_gamma, initial_beta, shots))
        else:
            initial_s = [np.random.uniform(0,self.mkp.knapsack_capacity[j]) for j in range(self.mkp.num_knapsacks)]
            initial_params = np.concatenate([initial_gamma, initial_beta, initial_s])
            print("Initial energy: ", self.compute_expectation(initial_gamma, initial_beta, shots, s=initial_s))
            
        # Objective function for classical optimizer
        def objective(params):
            gamma = params[:self.p]
            beta = params[self.p:2*self.p]
            if self.cont_slack:
                s = params[2*self.p:]
                return self.compute_expectation(gamma, beta, shots, s=s)
            else:
                return self.compute_expectation(gamma, beta, shots)
        
        # Optimize
        print(f"Optimizing QAOA (use_slack={self.use_slack}, p={self.p})...")
        result = minimize(objective, initial_params, method='COBYLA', 
                         options={'maxiter': maxiter})
        
        optimal_params = result.x
        energy = result.fun
        print("Energy after parameter tuning:", energy)
        optimal_gamma = optimal_params[:self.p]
        optimal_beta = optimal_params[self.p:2*self.p]

        # Get final solution
        qc = self.create_qaoa_circuit(optimal_gamma, optimal_beta)
        job = self.simulator.run(qc, shots=shots)
        probs = job.result().quasi_dists[0].binary_probabilities()
        counts = {b: shots*probs[b] for b in probs}

        # Find best solution
        best_bitstring = None
        best_value = np.inf
        for bitstring, count in counts.items():
            bitstring_reversed = bitstring[::-1]
            item_bitstring = bitstring_reversed[:self.num_qubits_items]
            
            value = self.mkp.evaluate_bitstring(item_bitstring, penalty_assignment=self.P_a, penalty_capacity=self.P_c)
            if value < best_value:
                best_value = value
                best_bitstring = item_bitstring
        
        return best_bitstring, best_value, optimal_params, counts, energy