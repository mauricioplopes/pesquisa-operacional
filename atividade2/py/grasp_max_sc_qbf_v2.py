"""
GRASP (Greedy Randomized Adaptive Search Procedure) Implementation for MAX-SC-QBF
MAX-SC-QBF: Maximization of Quadratic Binary Function with Set Cover constraints

Adapted for MO824/MC859 - Activity 2
Updated to read new input format
"""

import random
import copy
from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar, Optional, Set, Dict
import time
import sys

# Generic type for solution elements
E = TypeVar('E')


class Solution(Generic[E]):
    """Solution class that stores elements and cost"""
    
    def __init__(self, elements_or_solution = None, cost: float = None):
        if isinstance(elements_or_solution, Solution):
            # Copy constructor
            self.elements = copy.deepcopy(elements_or_solution.elements)
            self.cost = elements_or_solution.cost
        elif elements_or_solution is not None:
            # Constructor with elements and cost
            self.elements = elements_or_solution if elements_or_solution is not None else []
            self.cost = cost if cost is not None else float('inf')
        else:
            # Default constructor
            self.elements = []
            self.cost = float('inf')
    
    def add(self, element: E):
        """Add element to solution"""
        self.elements.append(element)
    
    def remove(self, element: E):
        """Remove element from solution"""
        if element in self.elements:
            self.elements.remove(element)
    
    def __contains__(self, element: E):
        """Check if element is in solution"""
        return element in self.elements
    
    def __iter__(self):
        """Make solution iterable"""
        return iter(self.elements)
    
    def __len__(self):
        """Get solution size"""
        return len(self.elements)
    
    def is_empty(self):
        """Check if solution is empty"""
        return len(self.elements) == 0
    
    def size(self):
        """Get solution size"""
        return len(self.elements)
    
    def __str__(self):
        return f"Solution: cost=[{self.cost}], size=[{len(self.elements)}], elements={self.elements}"


class Evaluator(ABC, Generic[E]):
    """Abstract evaluator interface for optimization problems"""
    
    @abstractmethod
    def get_domain_size(self) -> int:
        """Get the size of the problem domain"""
        pass
    
    @abstractmethod
    def evaluate(self, solution: Solution[E]) -> float:
        """Evaluate a complete solution"""
        pass
    
    @abstractmethod
    def evaluate_insertion_cost(self, element: E, solution: Solution[E]) -> float:
        """Evaluate cost of inserting an element into solution"""
        pass
    
    @abstractmethod
    def evaluate_removal_cost(self, element: E, solution: Solution[E]) -> float:
        """Evaluate cost of removing an element from solution"""
        pass
    
    @abstractmethod
    def evaluate_exchange_cost(self, elem_in: E, elem_out: E, solution: Solution[E]) -> float:
        """Evaluate cost of exchanging two elements"""
        pass
    
    @abstractmethod
    def is_feasible(self, solution: Solution[E]) -> bool:
        """Check if solution is feasible"""
        pass


class AbstractGRASP(ABC, Generic[E]):
    """Abstract GRASP metaheuristic class"""
    
    verbose = True
    
    def __init__(self, obj_function: Evaluator[E], alpha: float, iterations: int, 
                 construction_method: str = "standard", local_search_method: str = "first_improving"):
        self.obj_function = obj_function
        self.alpha = alpha
        self.iterations = iterations
        self.construction_method = construction_method
        self.local_search_method = local_search_method
        self.best_cost = float('-inf')  # Maximization problem
        self.cost = float('-inf')
        self.best_sol: Optional[Solution[E]] = None
        self.sol: Optional[Solution[E]] = None
        self.CL: List[E] = []
        self.RCL: List[E] = []
        random.seed(0)  # For reproducibility
    
    @abstractmethod
    def make_CL(self) -> List[E]:
        """Create the Candidate List"""
        pass
    
    @abstractmethod
    def make_RCL(self) -> List[E]:
        """Create the Restricted Candidate List"""
        pass
    
    @abstractmethod
    def update_CL(self):
        """Update the Candidate List according to current solution"""
        pass
    
    @abstractmethod
    def create_empty_sol(self) -> Solution[E]:
        """Create an empty solution"""
        pass
    
    @abstractmethod
    def local_search(self) -> Solution[E]:
        """Perform local search optimization"""
        pass
    
    def constructive_heuristic(self) -> Solution[E]:
        """GRASP constructive heuristic phase with different construction methods"""
        
        if self.construction_method == "random_plus_greedy":
            return self.random_plus_greedy_construction()
        elif self.construction_method == "sampled_greedy":
            return self.sampled_greedy_construction()
        else:  # standard
            return self.standard_constructive_heuristic()
    
    def standard_constructive_heuristic(self) -> Solution[E]:
        """Standard GRASP constructive heuristic"""
        
        self.CL = self.make_CL()
        self.RCL = self.make_RCL()
        self.sol = self.create_empty_sol()
        self.cost = float('-inf')
        
        # Main constructive loop
        while not self.constructive_stop_criteria():
            max_cost = float('-inf')
            min_cost = float('inf')
            self.cost = self.obj_function.evaluate(self.sol)
            self.update_CL()
            
            if not self.CL:  # No more candidates
                break
            
            # Explore all candidates to find min and max costs
            for c in self.CL:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost < min_cost:
                    min_cost = delta_cost
                if delta_cost > max_cost:
                    max_cost = delta_cost
            
            # Build RCL with candidates within alpha threshold
            self.RCL.clear()
            threshold = max_cost - self.alpha * (max_cost - min_cost)
            for c in self.CL:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost >= threshold:  # For maximization
                    self.RCL.append(c)
            
            # Choose random candidate from RCL
            if self.RCL:
                rnd_index = random.randint(0, len(self.RCL) - 1)
                in_cand = self.RCL[rnd_index]
                self.CL.remove(in_cand)
                self.sol.add(in_cand)
                self.obj_function.evaluate(self.sol)
        
        return self.sol
    
    def random_plus_greedy_construction(self) -> Solution[E]:
        """Random plus greedy construction method"""
        
        self.CL = self.make_CL()
        self.sol = self.create_empty_sol()
        
        # Phase 1: Random selection (30% of elements)
        num_random = max(1, int(0.3 * len(self.CL)))
        for _ in range(min(num_random, len(self.CL))):
            if not self.CL:
                break
            rnd_idx = random.randint(0, len(self.CL) - 1)
            selected = self.CL.pop(rnd_idx)
            self.sol.add(selected)
        
        # Phase 2: Greedy completion
        while not self.constructive_stop_criteria() and self.CL:
            self.update_CL()
            if not self.CL:
                break
                
            best_candidate = None
            best_cost = float('-inf')
            
            for c in self.CL:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost > best_cost:
                    best_cost = delta_cost
                    best_candidate = c
            
            if best_candidate:
                self.CL.remove(best_candidate)
                self.sol.add(best_candidate)
                self.obj_function.evaluate(self.sol)
        
        return self.sol
    
    def sampled_greedy_construction(self) -> Solution[E]:
        """Sampled greedy construction method"""
        
        self.CL = self.make_CL()
        self.sol = self.create_empty_sol()
        sample_size = max(2, int(0.5 * len(self.CL)))  # Sample 50% of candidates
        
        while not self.constructive_stop_criteria() and self.CL:
            self.update_CL()
            if not self.CL:
                break
            
            # Sample candidates
            sample_size_current = min(sample_size, len(self.CL))
            sampled_candidates = random.sample(self.CL, sample_size_current)
            
            # Find best among sampled
            best_candidate = None
            best_cost = float('-inf')
            
            for c in sampled_candidates:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost > best_cost:
                    best_cost = delta_cost
                    best_candidate = c
            
            if best_candidate:
                self.CL.remove(best_candidate)
                self.sol.add(best_candidate)
                self.obj_function.evaluate(self.sol)
        
        return self.sol
    
    def solve(self) -> Solution[E]:
        """Main GRASP procedure"""
        
        self.best_sol = self.create_empty_sol()
        
        for i in range(self.iterations):
            self.constructive_heuristic()
            
            # Ensure feasibility before local search
            if not self.obj_function.is_feasible(self.sol):
                self.make_feasible()
            
            self.local_search()
            
            if self.best_sol.cost < self.sol.cost:  # Maximization
                self.best_sol = Solution(self.sol)
                if self.verbose:
                    print(f"(Iter. {i}) BestSol = {self.best_sol}")
        
        return self.best_sol
    
    def constructive_stop_criteria(self) -> bool:
        """Stopping criteria for constructive heuristic"""
        return self.obj_function.is_feasible(self.sol)
    
    @abstractmethod
    def make_feasible(self):
        """Make solution feasible if needed"""
        pass


class QBF_SC(Evaluator[int]):
    """Quadratic Binary Function with Set Cover constraints"""
    
    def __init__(self, filename: str):
        self.size, self.A, self.subsets = self.read_input(filename)
        self.variables = [0.0] * self.size
    
    def read_input(self, filename: str) -> tuple[int, List[List[float]], Dict[int, Set[int]]]:
        """Read QBF-SC instance from file in the new format:
        <n> (número de variáveis binárias)
        <s1> <s2> ... <sn> (número de elementos cobertos por cada subconjunto)
        <lista de elementos cobertos por S1>
        <lista de elementos cobertos por S2>
        ...
        <lista de elementos cobertos por Sn>
        <a11> <a12> ... <a1n>
        <a22> ... <a2n>
        ...
        <ann>
        """
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        line_idx = 0
        
        # Parse number of binary variables (n)
        n = int(lines[line_idx])
        line_idx += 1
        
        # Parse subset sizes (s1, s2, ..., sn)
        subset_sizes = list(map(int, lines[line_idx].split()))
        line_idx += 1
        
        if len(subset_sizes) != n:
            raise ValueError(f"Expected {n} subset sizes, got {len(subset_sizes)}")
        
        # Parse subset elements
        subsets = {}
        for i in range(n):
            if subset_sizes[i] > 0:
                if line_idx >= len(lines):
                    raise ValueError(f"Missing elements for subset {i}")
                subset_elements = list(map(int, lines[line_idx].split()))
                if len(subset_elements) != subset_sizes[i]:
                    raise ValueError(f"Subset {i} should have {subset_sizes[i]} elements, got {len(subset_elements)}")
                # Convert to 0-based indexing (assuming input is 1-based)
                subsets[i] = set(elem - 1 for elem in subset_elements if elem > 0)
                line_idx += 1
            else:
                subsets[i] = set()  # Empty subset
        
        # Parse QBF matrix A (upper triangular format)
        A = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            if line_idx >= len(lines):
                raise ValueError(f"Missing matrix row {i}")
            
            # Read elements from position i to n-1 (upper triangular)
            row_elements = list(map(float, lines[line_idx].split()))
            expected_elements = n - i
            
            if len(row_elements) != expected_elements:
                raise ValueError(f"Row {i} should have {expected_elements} elements, got {len(row_elements)}")
            
            # Fill upper triangular part
            for j, val in enumerate(row_elements):
                col_idx = i + j
                if col_idx < n:
                    A[i][col_idx] = val
                    # Make matrix symmetric for easier computation
                    if col_idx != i:
                        A[col_idx][i] = val
            
            line_idx += 1
        
        return n, A, subsets
    
    def get_domain_size(self) -> int:
        return self.size
    
    def set_variables(self, solution: Solution[int]):
        """Set binary variables based on solution"""
        self.reset_variables()
        if not solution.is_empty():
            for elem in solution:
                if 0 <= elem < self.size:
                    self.variables[elem] = 1.0
    
    def reset_variables(self):
        """Reset all variables to 0"""
        self.variables = [0.0] * self.size
    
    def evaluate(self, solution: Solution[int]) -> float:
        """Evaluate QBF: f(x) = x'.A.x"""
        self.set_variables(solution)
        
        if not self.is_feasible(solution):
            solution.cost = float('-inf')  # Infeasible solution
            return solution.cost
        
        solution.cost = self.evaluate_QBF()
        return solution.cost
    
    def evaluate_QBF(self) -> float:
        """Matrix multiplication for QBF evaluation"""
        total = 0.0
        
        for i in range(self.size):
            for j in range(self.size):
                total += self.variables[i] * self.variables[j] * self.A[i][j]
        
        return total
    
    def is_feasible(self, solution: Solution[int]) -> bool:
        """Check if solution satisfies set cover constraints"""
        covered_variables = set()
        
        for subset_idx in solution:
            if subset_idx in self.subsets:
                covered_variables.update(self.subsets[subset_idx])
        
        # All variables must be covered (0 to size-1)
        required_coverage = set(range(self.size))
        return covered_variables == required_coverage
    
    def get_uncovered_variables(self, solution: Solution[int]) -> Set[int]:
        """Get variables not covered by current solution"""
        covered_variables = set()
        
        for subset_idx in solution:
            if subset_idx in self.subsets:
                covered_variables.update(self.subsets[subset_idx])
        
        all_variables = set(range(self.size))
        return all_variables - covered_variables
    
    def evaluate_insertion_cost(self, elem: int, solution: Solution[int]) -> float:
        if elem in solution:
            return 0.0
        
        self.set_variables(solution)
        return self.evaluate_insertion_QBF(elem)
    
    def evaluate_insertion_QBF(self, i: int) -> float:
        """Evaluate insertion cost efficiently"""
        if self.variables[i] == 1:
            return 0.0
        return self.evaluate_contribution_QBF(i)
    
    def evaluate_removal_cost(self, elem: int, solution: Solution[int]) -> float:
        if elem not in solution:
            return 0.0
        
        # Check if removal maintains feasibility
        temp_sol = Solution(list(solution.elements))
        temp_sol.remove(elem)
        if not self.is_feasible(temp_sol):
            return float('-inf')  # Cannot remove - would make infeasible
        
        self.set_variables(solution)
        return self.evaluate_removal_QBF(elem)
    
    def evaluate_removal_QBF(self, i: int) -> float:
        """Evaluate removal cost efficiently"""
        if self.variables[i] == 0:
            return 0.0
        return -self.evaluate_contribution_QBF(i)
    
    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, solution: Solution[int]) -> float:
        # Check if exchange maintains feasibility
        temp_sol = Solution(list(solution.elements))
        temp_sol.remove(elem_out)
        temp_sol.add(elem_in)
        if not self.is_feasible(temp_sol):
            return float('-inf')  # Cannot exchange - would make infeasible
        
        self.set_variables(solution)
        return self.evaluate_exchange_QBF(elem_in, elem_out)
    
    def evaluate_exchange_QBF(self, elem_in: int, elem_out: int) -> float:
        """Evaluate exchange cost efficiently"""
        if elem_in == elem_out:
            return 0.0
        if self.variables[elem_in] == 1:
            return self.evaluate_removal_QBF(elem_out)
        if self.variables[elem_out] == 0:
            return self.evaluate_insertion_QBF(elem_in)
        
        cost = 0.0
        cost += self.evaluate_contribution_QBF(elem_in)
        cost -= self.evaluate_contribution_QBF(elem_out)
        cost -= (self.A[elem_in][elem_out] + self.A[elem_out][elem_in])
        
        return cost
    
    def evaluate_contribution_QBF(self, i: int) -> float:
        """Evaluate contribution of element i to QBF"""
        total = 0.0
        
        for j in range(self.size):
            if i != j:
                total += self.variables[j] * (self.A[i][j] + self.A[j][i])
        total += self.A[i][i]
        
        return total


class GRASP_QBF_SC(AbstractGRASP[int]):
    """GRASP implementation for QBF with Set Cover constraints"""
    
    def __init__(self, alpha: float, iterations: int, filename: str, 
                 construction_method: str = "standard", local_search_method: str = "first_improving"):
        super().__init__(QBF_SC(filename), alpha, iterations, construction_method, local_search_method)
    
    def make_CL(self) -> List[int]:
        """Create candidate list with all subsets"""
        return list(range(self.obj_function.get_domain_size()))
    
    def make_RCL(self) -> List[int]:
        """Create empty RCL (populated during constructive phase)"""
        return []
    
    def update_CL(self):
        """Update CL to include only subsets that can cover uncovered variables"""
        if self.obj_function.is_feasible(self.sol):
            # If already feasible, any subset not in solution can be candidate
            self.CL = [i for i in range(self.obj_function.get_domain_size()) if i not in self.sol]
        else:
            # Only consider subsets that cover uncovered variables
            uncovered = self.obj_function.get_uncovered_variables(self.sol)
            self.CL = []
            for i in range(self.obj_function.get_domain_size()):
                if i not in self.sol:
                    subset_coverage = self.obj_function.subsets.get(i, set())
                    if uncovered.intersection(subset_coverage):  # Covers at least one uncovered variable
                        self.CL.append(i)
    
    def create_empty_sol(self) -> Solution[int]:
        """Create empty solution"""
        sol = Solution[int]()
        sol.cost = float('-inf')
        return sol
    
    def make_feasible(self):
        """Make solution feasible by adding necessary subsets"""
        max_attempts = 100
        attempts = 0
        
        while not self.obj_function.is_feasible(self.sol) and attempts < max_attempts:
            uncovered = self.obj_function.get_uncovered_variables(self.sol)
            if not uncovered:
                break
            
            # Find subset that covers most uncovered variables
            best_subset = None
            best_coverage = 0
            
            for i in range(self.obj_function.get_domain_size()):
                if i not in self.sol:
                    subset_coverage = self.obj_function.subsets.get(i, set())
                    coverage_count = len(uncovered.intersection(subset_coverage))
                    if coverage_count > best_coverage:
                        best_coverage = coverage_count
                        best_subset = i
            
            if best_subset is not None:
                self.sol.add(best_subset)
            else:
                break
            
            attempts += 1
        
        self.obj_function.evaluate(self.sol)
    
    def local_search(self) -> Solution[int]:
        """Local search with insertion, removal, and exchange moves"""
        
        improved = True
        while improved:
            improved = False
            best_move = None
            best_delta = 0.0
            
            # Current candidates (not in solution)
            current_CL = [i for i in range(self.obj_function.get_domain_size()) if i not in self.sol]
            
            # Evaluate insertions
            for cand_in in current_CL:
                delta_cost = self.obj_function.evaluate_insertion_cost(cand_in, self.sol)
                if delta_cost > best_delta or (self.local_search_method == "first_improving" and delta_cost > 0):
                    best_delta = delta_cost
                    best_move = ("insert", cand_in, None)
                    if self.local_search_method == "first_improving" and delta_cost > 0:
                        break
            
            if not (self.local_search_method == "first_improving" and best_move):
                # Evaluate removals
                for cand_out in list(self.sol):
                    delta_cost = self.obj_function.evaluate_removal_cost(cand_out, self.sol)
                    if delta_cost > best_delta or (self.local_search_method == "first_improving" and delta_cost > 0):
                        best_delta = delta_cost
                        best_move = ("remove", None, cand_out)
                        if self.local_search_method == "first_improving" and delta_cost > 0:
                            break
            
            if not (self.local_search_method == "first_improving" and best_move):
                # Evaluate exchanges
                for cand_in in current_CL:
                    for cand_out in list(self.sol):
                        delta_cost = self.obj_function.evaluate_exchange_cost(cand_in, cand_out, self.sol)
                        if delta_cost > best_delta or (self.local_search_method == "first_improving" and delta_cost > 0):
                            best_delta = delta_cost
                            best_move = ("exchange", cand_in, cand_out)
                            if self.local_search_method == "first_improving" and delta_cost > 0:
                                break
                    if self.local_search_method == "first_improving" and best_move and best_move[0] == "exchange":
                        break
            
            # Apply best move
            if best_move and best_delta > 1e-10:
                move_type, cand_in, cand_out = best_move
                
                if move_type == "insert" and cand_in is not None:
                    self.sol.add(cand_in)
                elif move_type == "remove" and cand_out is not None:
                    self.sol.remove(cand_out)
                elif move_type == "exchange" and cand_in is not None and cand_out is not None:
                    self.sol.remove(cand_out)
                    self.sol.add(cand_in)
                
                self.obj_function.evaluate(self.sol)
                improved = True
        
        return self.sol


def create_sample_instance():
    """Create a sample QBF-SC instance for testing in the new format"""
    content = """4
2 3 2 1
1 2
2 3 4
1 4
3
10 -2 3 1
5 0 -1
8 4
-2"""
    
    with open('sample_qbf_sc_new.txt', 'w') as f:
        f.write(content)
    
    return 'sample_qbf_sc_new.txt'


def run_experiments():
    """Run computational experiments as specified in the activity"""
    
    filename = create_sample_instance()
    results = []
    
    # Configuration parameters
    alphas = [0.1, 0.3]  # α₁ and α₂
    constructions = ["standard", "random_plus_greedy", "sampled_greedy"]
    local_searches = ["first_improving", "best_improving"]
    iterations = 100  # Reduced for testing
    
    print("Running computational experiments...")
    print("=" * 60)
    
    # Base configuration
    alpha1 = alphas[0]
    
    configs = [
        ("PADRÃO", alpha1, "first_improving", "standard"),
        ("PADRÃO+ALPHA", alphas[1], "first_improving", "standard"),
        ("PADRÃO+BEST", alpha1, "best_improving", "standard"),
        ("PADRÃO+HC1", alpha1, "first_improving", "random_plus_greedy"),
        ("PADRÃO+HC2", alpha1, "first_improving", "sampled_greedy"),
    ]
    
    for config_name, alpha, local_search, construction in configs:
        print(f"\nRunning {config_name}...")
        start_time = time.time()
        
        grasp = GRASP_QBF_SC(
            alpha=alpha, 
            iterations=iterations, 
            filename=filename,
            construction_method=construction,
            local_search_method=local_search
        )
        
        best_sol = grasp.solve()
        end_time = time.time()
        
        results.append({
            'config': config_name,
            'cost': best_sol.cost,
            'size': len(best_sol),
            'time': end_time - start_time,
            'feasible': grasp.obj_function.is_feasible(best_sol)
        })
        
        print(f"Cost: {best_sol.cost}, Size: {len(best_sol)}, Time: {end_time - start_time:.2f}s")
    
    # Print results table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<15} {'Cost':<10} {'Size':<6} {'Time(s)':<8} {'Feasible':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['config']:<15} {result['cost']:<10.2f} {result['size']:<6} "
              f"{result['time']:<8.2f} {result['feasible']:<10}")
    
    return results


def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--experiment":
            run_experiments()
            return
        
        filename = sys.argv[1]
        alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        construction = sys.argv[4] if len(sys.argv) > 4 else "standard"
        local_search = sys.argv[5] if len(sys.argv) > 5 else "first_improving"
    else:
        # Default test
        filename = create_sample_instance()
        alpha = 0.1
        iterations = 50
        construction = "standard"
        local_search = "first_improving"
    
    print(f"Running GRASP for MAX-SC-QBF")
    print(f"File: {filename}")
    print(f"Alpha: {alpha}, Iterations: {iterations}")
    print(f"Construction: {construction}, Local Search: {local_search}")
    print("-" * 60)
    
    start_time = time.time()
    grasp = GRASP_QBF_SC(
        alpha=alpha, 
        iterations=iterations, 
        filename=filename,
        construction_method=construction,
        local_search_method=local_search
    )
    
    best_sol = grasp.solve()
    end_time = time.time()
    
    print(f"\nFinal Result:")
    print(f"Best Solution: {best_sol}")
    print(f"Feasible: {grasp.obj_function.is_feasible(best_sol)}")
    print(f"Execution Time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    main()