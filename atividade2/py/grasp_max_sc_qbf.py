import random
import copy
import sys
import time
from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar, Optional, Set, Dict

# Generic type for solution elements
E = TypeVar('E')

class Solution(Generic[E]):
    """Solution class that extends a list to store elements and cost"""
    def __init__(self, elements_or_solution=None, cost: float = None):
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

# --- Ajustes para o problema MAX-SC-QBF começam aqui ---

# Define um peso alto para a cobertura de elementos para garantir a prioridade da factibilidade [3]
# Este valor deve ser grande o suficiente para dominar qualquer ganho/perda potencial da função QBF.
# Por exemplo, para n=400 e coeficientes A entre -10 e 10, o valor máximo da QBF é da ordem de 1.6 * 10^6.
# Um peso de 10^8 garante que cobrir um elemento descoberto é sempre mais vantajoso.
WEIGHT_COVERAGE = 1e8 

class QBF(Evaluator[int]):
    """Implementação da Função Binária Quadrática (QBF)
    Adaptada para ler dados de Set Cover e gerenciar o estado de cobertura.
    """
    def __init__(self, filename: str):
        # self.filename_for_init é útil para criar avaliadores temporários em evaluate_exchange_cost.
        self.filename_for_init = filename
        self.size, self.subsets_covered_elements, self.A = self.read_input(filename)
        self.variables = [0.0] * self.size # Variáveis binárias x_i (0 ou 1)

        # Mapeia elementos para os subconjuntos que os cobrem para busca eficiente nas checagens de Set Cover
        # element_to_covering_subsets[k] = lista de índices de subconjuntos que cobrem o elemento k
        self.element_to_covering_subsets: Dict[int, List[int]] = {k: [] for k in range(self.size)}
        for s_idx, covered_by_s in enumerate(self.subsets_covered_elements):
            for k_element in covered_by_s:
                self.element_to_covering_subsets[k_element].append(s_idx)

        # Estado para as restrições de Set Cover
        self.current_covered_elements: Set[int] = set()
        self.num_uncovered_elements: int = self.size # Inicialmente, todos os elementos estão descobertos

    def read_input(self, filename: str) -> tuple[int, List[List[int]], List[List[float]]]:
        """Lê os dados da instância MAX-SC-QBF: n, listas de elementos cobertos pelos subconjuntos e matriz A."""
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

            if not lines:
                raise ValueError("Arquivo vazio ou inválido")

            # 1. Lê o tamanho (n) [4]
            size = int(lines[0])

            line_idx = 1
            
            # Check if this is a standard QBF format (matrix only) or Set Cover format
            # Standard QBF format: just the matrix data after size
            # Set Cover format: subset counts, then subsets, then matrix
            
            # Try to determine format by checking if second line contains subset counts or matrix data
            if line_idx < len(lines):
                second_line_values = list(map(int, lines[line_idx].split()))
                # If second line has exactly 'size' elements and they're all positive small numbers,
                # it's likely subset counts. Otherwise, it's matrix data.
                if len(second_line_values) == size and all(v > 0 and v <= size for v in second_line_values):
                    # Set Cover format
                    # 2. Ignora a linha de contagens de elementos por subconjunto (presente no formato, mas não usada diretamente) [5]
                    _ = lines[line_idx]
                    line_idx += 1

                    # 3. Lê as listas de elementos cobertos por cada subconjunto [4, 5]
                    subsets_covered_elements: List[List[int]] = []
                    for i in range(size):
                        if line_idx >= len(lines):
                            raise ValueError(f"Arquivo incompleto: esperado {size} linhas de subconjuntos, encontrado apenas {i}")
                        line_elements = list(map(int, lines[line_idx].split()))
                        # Ajusta os elementos para 0-indexados internamente, pois a entrada é 1-indexada [6]
                        subsets_covered_elements.append([e - 1 for e in line_elements if e > 0])
                        line_idx += 1
                else:
                    # Standard QBF format - create dummy subsets where each subset covers only itself
                    subsets_covered_elements: List[List[int]] = [[i] for i in range(size)]

            # 4. Inicializa a matriz A (todos os elementos 0.0)
            A = [[0.0] * size for _ in range(size)]

            # 5. Lê a matriz A (formato triangular superior) [4, 7]
            # O código original em grasp_python_3.txt preenche A[i][j] e A[j][i] = 0.0 se j > i.
            # Isso cria uma matriz assimétrica. A função evaluate_contribution_QBF [8]
            # soma (A[i][j] + A[j][i]). Mantemos essa lógica do arquivo original.
            for i in range(size):
                if line_idx >= len(lines):
                    raise ValueError(f"Arquivo incompleto: esperado dados da matriz na linha {line_idx + 1}")
                values = list(map(float, lines[line_idx].split()))
                for j_offset, val in enumerate(values):
                    j = i + j_offset
                    if j < size:  # Boundary check
                        A[i][j] = val
                        if i != j: # Para elementos fora da diagonal
                            A[j][i] = 0.0 # Define a parte inferior como 0, conforme o comportamento original [9]
                line_idx += 1
            
            return size, subsets_covered_elements, A

    def _update_coverage_state(self, solution: Solution[int]):
        """
        Atualiza o estado interno de elementos cobertos e a contagem de elementos descobertos
        com base na solução parcial `solution`.
        """
        self.current_covered_elements.clear()
        
        # Constrói um conjunto temporário de elementos cobertos pela solução atual
        temp_covered = set()
        for s_idx in solution.elements:
            for k_element in self.subsets_covered_elements[s_idx]:
                temp_covered.add(k_element)
        
        self.current_covered_elements = temp_covered
        self.num_uncovered_elements = self.size - len(self.current_covered_elements)

    def _is_set_cover_feasible(self) -> bool:
        """
        Verifica se o estado interno atual (self.current_covered_elements)
        representa uma cobertura de conjuntos factível (todos os elementos estão cobertos).
        """
        return self.num_uncovered_elements == 0

    def _count_new_covered_by_insertion(self, elem_to_insert: int, solution: Solution[int]) -> int:
        """
        Calcula quantos *novos* elementos seriam cobertos se 'elem_to_insert' fosse adicionado à solução.
        """
        newly_covered_count = 0
        # Verifica se elem_to_insert (um índice de subconjunto) realmente cobre elementos
        if 0 <= elem_to_insert < len(self.subsets_covered_elements):
            for k_element in self.subsets_covered_elements[elem_to_insert]:
                if k_element not in self.current_covered_elements: # Se o elemento ainda não está coberto
                    newly_covered_count += 1
        return newly_covered_count

    def _would_removal_be_feasible(self, elem_to_remove: int, current_solution_elements_list: List[int]) -> bool:
        """
        Verifica se a remoção de `elem_to_remove` manteria a factibilidade da cobertura de conjuntos.
        Um subconjunto só pode ser removido se não for o *único* a cobrir qualquer um dos elementos que ele cobre [3].
        `current_solution_elements_list` é uma lista de índices dos subconjuntos selecionados (x_i = 1).
        """
        if elem_to_remove not in current_solution_elements_list:
            return True # Se não está na solução, não pode ser removido, então é "factível" neste contexto.

        # Itera sobre os elementos cobertos por 'elem_to_remove'
        elements_covered_by_removed = self.subsets_covered_elements[elem_to_remove]
        
        for k_element in elements_covered_by_removed:
            is_covered_by_another_subset = False
            # Verifica outros subconjuntos selecionados (excluindo elem_to_remove)
            for other_s_idx in current_solution_elements_list:
                if other_s_idx != elem_to_remove:
                    if k_element in self.subsets_covered_elements[other_s_idx]:
                        is_covered_by_another_subset = True
                        break # Encontrou outro subconjunto cobrindo k_element
            
            if not is_covered_by_another_subset:
                # Se k_element é *unicamente* coberto por elem_to_remove, a remoção é inviável
                return False
        
        return True # Todos os elementos cobertos por elem_to_remove ainda seriam cobertos

    def set_variables(self, solution: Solution[int]):
        """Define as variáveis binárias com base na solução e atualiza o estado de cobertura."""
        # A chamada original set_variables da classe base QBF.
        self.reset_variables() # Zera todas as variáveis primeiro
        if not solution.is_empty():
            for elem in solution:
                self.variables[elem] = 1.0
        # Atualiza o estado específico do Set Cover
        self._update_coverage_state(solution)

    def reset_variables(self):
        """Redefine todas as variáveis para 0 e o estado de cobertura."""
        self.variables = [0.0] * self.size
        self.current_covered_elements.clear()
        self.num_uncovered_elements = self.size # Todos os elementos estão descobertos

    def get_domain_size(self) -> int:
        return self.size

    def evaluate_QBF(self) -> float:
        """Avaliação direta da função QBF (f(x) = x'Ax) para o estado atual das variáveis."""
        total = 0.0
        vec_aux = [0.0] * self.size
        for i in range(self.size):
            aux = 0.0
            for j in range(self.size):
                aux += self.variables[j] * self.A[i][j]
            vec_aux[i] = aux
        for i in range(self.size): # Este loop estava faltando na versão original, somando os termos x_i * aux
            total += vec_aux[i] * self.variables[i] # Multiplica o resultado do produto interno pelo x_i correspondente
        return total

    def evaluate_contribution_QBF(self, i: int) -> float:
        """Avalia a contribuição de um elemento 'i' à QBF se adicionado ou removido."""
        total = 0.0
        for j in range(self.size):
            if i != j:
                # Soma A[i][j]*x_j + A[j][i]*x_j.
                # Se A[j][i] for 0 (como na leitura original), contribui apenas A[i][j]*x_j.
                total += self.variables[j] * (self.A[i][j] + self.A[j][i])
        total += self.A[i][i] # Termo quadrático x_i * x_i = x_i
        return total

    def evaluate_removal_QBF(self, i: int) -> float:
        """Avalia o custo QBF de remover um elemento 'i' (valor positivo para perda, negativo para ganho)."""
        # A remoção inverte a contribuição da inserção.
        return -self.evaluate_contribution_QBF(i)

    def evaluate_exchange_QBF(self, elem_in: int, elem_out: int) -> float:
        """Avalia o custo QBF de trocar 'elem_out' por 'elem_in'.
        Retorna a mudança líquida no valor da QBF.
        """
        if elem_in == elem_out:
            return 0.0

        # Calcula a contribuição de 'elem_in'
        gain_in = self.evaluate_contribution_QBF(elem_in)
        # Calcula a contribuição de 'elem_out'
        loss_out = self.evaluate_contribution_QBF(elem_out)

        # Ajusta para a interação entre elem_in e elem_out
        interaction_term = (self.A[elem_in][elem_out] + self.A[elem_out][elem_in])
        
        # A mudança líquida é o ganho do 'in' menos a perda do 'out', menos a interação que seria contada duas vezes
        return gain_in - loss_out - interaction_term

    def print_matrix(self):
        """Imprime a matriz QBF"""
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append(str(self.A[i][j]))
            print(" ".join(row))

class QBF_Inverse(QBF):
    """QBF Inversa para problemas de maximização (GRASP minimiza por padrão).
    Modificada para incorporar as restrições de Set Cover nos custos de avaliação.
    """
    def __init__(self, filename: str):
        super().__init__(filename)
        # self.filename_for_init é herdado e já está definido no __init__ de QBF.

    def evaluate(self, solution: Solution[int]) -> float:
        """Avalia uma solução completa para MAX-SC-QBF. Retorna -valor_objetivo."""
        self.set_variables(solution) # Atualiza o estado do avaliador para a solução fornecida
        
        if not self._is_set_cover_feasible():
            # Se a solução não for factível, atribui um custo muito alto (infinito para minimização)
            solution.cost = float('inf') 
        else:
            # Se for factível, calcula o valor QBF e o nega para converter em problema de minimização
            solution.cost = -super().evaluate_QBF() 
        return solution.cost

    def evaluate_insertion_cost(self, element: int, solution: Solution[int]) -> float:
        """Avalia o custo de inserção de um elemento para MAX-SC-QBF."""
        self.set_variables(solution) # Garante que o estado reflita a solução atual
        
        if self.variables[element] == 1:
            return 0.0 # Elemento já na solução, nenhum custo de inserção
        
        # Calcula o ganho QBF potencial (usando o cálculo da classe base QBF)
        qbf_gain = super().evaluate_contribution_QBF(element)

        # Calcula quantos *novos* elementos seriam cobertos
        new_elements_covered_count = self._count_new_covered_by_insertion(element, solution)
        
        coverage_bonus = 0.0
        # Aplica o peso alto apenas se ainda houver elementos descobertos [3]
        if self.num_uncovered_elements > 0: 
            coverage_bonus = WEIGHT_COVERAGE * new_elements_covered_count
        
        # A pontuação gananciosa (greedy_score) combina ganho QBF com bônus de cobertura
        greedy_score = qbf_gain + coverage_bonus
        
        # Retorna o negativo da pontuação gananciosa, pois QBF_Inverse minimiza (menor custo = melhor)
        return -greedy_score

    def evaluate_removal_cost(self, element: int, solution: Solution[int]) -> float:
        """Avalia o custo de remoção de um elemento para MAX-SC-QBF."""
        self.set_variables(solution) # Garante que o estado reflita a solução atual
        
        if self.variables[element] == 0:
            return 0.0 # Elemento não está na solução
            
        current_sol_elements_list = [idx for idx, val in enumerate(self.variables) if val == 1]
        # Verifica a factibilidade da remoção em relação às restrições de Set Cover [3]
        if not self._would_removal_be_feasible(element, current_sol_elements_list):
            return float('inf') # Remoção inviável, atribui custo muito alto
        
        # Calcula a mudança QBF (perda) da remoção (usando o cálculo da classe base QBF)
        qbf_loss = super().evaluate_removal_QBF(element) # Isso já é -(contribuição_do_elemento)
        
        # Retorna o negativo da perda QBF para a minimização (uma perda positiva se torna um custo positivo)
        return -qbf_loss

    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, solution: Solution[int]) -> float:
        """Avalia o custo de troca de dois elementos para MAX-SC-QBF."""
        self.set_variables(solution) # Garante que o estado reflita a solução atual
        
        if elem_in == elem_out:
            return 0.0

        # Constrói uma solução hipotética para verificar a factibilidade da cobertura
        current_sol_elements_list = [idx for idx, val in enumerate(self.variables) if val == 1]
        hypothetical_sol_elements_list = [e for e in current_sol_elements_list if e != elem_out]
        if elem_in not in hypothetical_sol_elements_list: # Adiciona apenas se não estiver presente
            hypothetical_sol_elements_list.append(elem_in)
        
        # Verifica a factibilidade usando uma instância temporária de QBF_Inverse
        # Isso evita modificar o estado do avaliador atual durante a verificação.
        temp_qbf_evaluator = QBF_Inverse(self.filename_for_init) 
        temp_qbf_evaluator.set_variables(Solution(hypothetical_sol_elements_list))
        
        if not temp_qbf_evaluator._is_set_cover_feasible():
            return float('inf') # Troca inviável, atribui custo muito alto
            
        # Se factível, calcula a mudança QBF (usando o cálculo da classe base QBF)
        qbf_change = super().evaluate_exchange_QBF(elem_in, elem_out)
        
        # Retorna o negativo da mudança QBF para a minimização
        return -qbf_change

class AbstractGRASP(ABC, Generic[E]):
    """Classe abstrata para a meta-heurística GRASP"""
    verbose = True

    def __init__(self, obj_function: Evaluator[E], alpha: float, iterations: int):
        self.obj_function = obj_function
        self.alpha = alpha
        self.iterations = iterations
        self.best_cost = float('inf')
        self.cost = float('inf')
        self.best_sol: Optional[Solution[E]] = None
        self.sol: Optional[Solution[E]] = None
        self.CL: List[E] = []
        self.RCL: List[E] = []

        random.seed(0) # Para reprodutibilidade

    @abstractmethod
    def make_CL(self) -> List[E]:
        """Cria a Lista de Candidatos (Candidate List)"""
        pass

    @abstractmethod
    def make_RCL(self) -> List[E]:
        """Cria a Lista Restrita de Candidatos (Restricted Candidate List)"""
        pass

    @abstractmethod
    def update_CL(self):
        """Atualiza a Lista de Candidatos de acordo com a solução atual"""
        pass

    @abstractmethod
    def create_empty_sol(self) -> Solution[E]:
        """Cria uma solução vazia"""
        pass

    @abstractmethod
    def local_search(self) -> Solution[E]:
        """Executa a fase de busca local"""
        pass

    def constructive_heuristic(self) -> Solution[E]:
        """Fase de heurística construtiva do GRASP"""
        self.CL = self.make_CL()
        self.RCL = self.make_RCL()
        self.sol = self.create_empty_sol()
        self.cost = float('inf') # Custo inicial, será atualizado pelo evaluate

        # Laço principal da construção
        # A condição de parada foi modificada na classe GRASP_QBF para o Set Cover [3]
        while not self.constructive_stop_criteria():
            max_cost = float('-inf')
            min_cost = float('inf')
            
            self.update_CL() # Atualiza a lista de candidatos (elementos não na solução)

            if not self.CL: # Se a lista de candidatos estiver vazia e os critérios não forem atendidos, encerra
                break

            # Explora todos os candidatos para encontrar os custos mínimo e máximo para formar a RCL
            for c in self.CL:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost < min_cost:
                    min_cost = delta_cost
                if delta_cost > max_cost:
                    max_cost = delta_cost
            
            # Constrói a RCL com candidatos dentro do limiar alpha [10, 11]
            self.RCL.clear()
            for c in self.CL:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                # Para maximização (QBF_Inverse minimiza), um delta_cost menor (mais negativo) é melhor.
                # O limiar deve ser relativo ao min_cost.
                if delta_cost <= min_cost + self.alpha * (max_cost - min_cost):
                    self.RCL.append(c)

            # Escolhe um candidato aleatório da RCL [10]
            if self.RCL:
                rnd_index = random.randint(0, len(self.RCL) - 1)
                in_cand = self.RCL[rnd_index]
                self.CL.remove(in_cand) # Remove da CL uma vez adicionado à solução
                self.sol.add(in_cand)
                # Atualiza o custo real da solução, o que também atualiza o estado de cobertura interno do avaliador
                self.obj_function.evaluate(self.sol) 
                self.cost = self.sol.cost # Atualiza o custo corrente da solução construída
                self.RCL.clear()
            else:
                # Se a RCL estiver vazia mas a CL não, significa que nenhum candidato atendeu ao critério alpha.
                # Isso pode indicar que todos os movimentos restantes são ruins ou que alpha é muito restritivo.
                # Para evitar loops infinitos, paramos a construção.
                break 

        # Após a construção, verifica se a solução é factível para o Set Cover [3]
        if not self.obj_function._is_set_cover_feasible():
            # Se não for factível, a iteração é descartada atribuindo um custo muito alto [3]
            self.sol.cost = float('inf') 
            if self.verbose:
                print("Solução construída é inviável para Set Cover. Descartando iteração.")
        
        return self.sol

    def solve(self) -> Solution[E]:
        """Procedimento principal do GRASP"""
        self.best_sol = self.create_empty_sol()
        # Inicializa best_sol com o valor mais negativo possível para MAX-QBF (infinito para minimização de -QBF)
        self.best_sol.cost = float('inf') 

        for i in range(self.iterations):
            # Gera uma nova semente aleatória para cada iteração para garantir diversificação [12]
            random.seed(random.randint(0, 1000000) + i) 

            self.constructive_heuristic()
            
            # Procede para a busca local apenas se a solução construída for factível
            if self.sol.cost != float('inf'): # Verifica se a solução não foi descartada
                self.local_search()

            # Atualiza a melhor solução global se a solução atual for melhor
            if self.sol.cost < self.best_sol.cost: # Para minimização, custo menor é melhor
                self.best_sol = Solution(self.sol) # Cria uma cópia profunda da solução
            
            if self.verbose:
                # Imprime o custo da melhor solução da iteração atual e da melhor solução global
                print(f"(Iter. {i+1}) Custo Sol. Atual = {self.sol.cost:.4f}, Melhor Custo Global = {self.best_sol.cost:.4f}")

        return self.best_sol

    def constructive_stop_criteria(self) -> bool:
        """Critério de parada para a heurística construtiva para MAX-SC-QBF."""
        # A construção para quando todos os elementos estão cobertos [3] OU quando a CL está vazia.
        return self.obj_function._is_set_cover_feasible() or not self.CL

class GRASP_QBF(AbstractGRASP[int]):
    """Implementação GRASP para o problema MAX-SC-QBF."""

    def __init__(self, alpha: float, iterations: int, filename: str):
        super().__init__(QBF_Inverse(filename), alpha, iterations)
        # Garante que o estado inicial do obj_function (avaliador) esteja consistente
        self.obj_function.reset_variables() 

    def make_CL(self) -> List[int]:
        """Cria a lista de candidatos com todos os elementos do domínio (índices de subconjuntos/variáveis)."""
        # Inicialmente, todos os subconjuntos são candidatos a serem adicionados
        return list(range(self.obj_function.get_domain_size()))

    def make_RCL(self) -> List[int]:
        """Cria uma RCL vazia (preenchida durante a fase construtiva)."""
        return []

    def update_CL(self):
        """
        Atualiza a CL - para MAX-SC-QBF, a CL contém os elementos que ainda NÃO estão na solução.
        A lógica de remover elementos da CL à medida que são adicionados à solução já é tratada
        em `constructive_heuristic`, então este método não precisa de lógica adicional aqui.
        """
        pass

    def create_empty_sol(self) -> Solution[int]:
        """Cria uma solução vazia com custo zero (que é -0.0 para QBF_Inverse)."""
        sol = Solution[int]()
        sol.cost = 0.0 # Custo de uma solução vazia (0 para MAX-QBF, então 0 para QBF_Inverse)
        return sol

    def local_search(self) -> Solution[int]:
        """Busca local com movimentos 1-flip (inserção, remoção e troca).
        Para MAX-SC-QBF, a busca local deve respeitar a factibilidade de Set Cover,
        especialmente durante a remoção e troca. Os métodos `evaluate_removal_cost`
        e `evaluate_exchange_cost` em QBF_Inverse já lidam com isso, retornando
        `float('inf')` para movimentos inviáveis.
        """
        while True:
            min_delta_cost = float('inf') # Menor (mais negativo) custo de melhoria
            best_cand_in = None
            best_cand_out = None

            # Constrói a lista atual de elementos que podem ser inseridos (aqueles não na solução)
            current_CL_not_in_sol = []
            for i in range(self.obj_function.get_domain_size()):
                if i not in self.sol:
                    current_CL_not_in_sol.append(i)

            # Avalia movimentos 1-flip:

            # 1. Avalia inserções (0 -> 1)
            # Uma virada de 0 para 1 é sempre factível em relação à cobertura [3].
            # É considerada apenas se o ganho na função objetivo for positivo (delta_cost < 0).
            for cand_in in current_CL_not_in_sol:
                delta_cost = self.obj_function.evaluate_insertion_cost(cand_in, self.sol)
                if delta_cost < min_delta_cost: # Para minimização, mais negativo é melhor
                    min_delta_cost = delta_cost
                    best_cand_in = cand_in
                    best_cand_out = None # Este é um movimento de inserção

            # 2. Avalia remoções (1 -> 0)
            # É preciso verificar a factibilidade: `_would_removal_be_feasible` [3].
            # Se a remoção tornar a solução inviável, evaluate_removal_cost retornará float('inf'),
            # garantindo que o movimento não seja escolhido.
            for cand_out in self.sol:
                delta_cost = self.obj_function.evaluate_removal_cost(cand_out, self.sol)
                if delta_cost < min_delta_cost:
                    min_delta_cost = delta_cost
                    best_cand_in = None # Este é um movimento de remoção
                    best_cand_out = cand_out
            
            # 3. Avalia trocas (1 -> 0 e 0 -> 1 simultaneamente)
            # Também é preciso verificar a factibilidade (tratado por evaluate_exchange_cost).
            for cand_in in current_CL_not_in_sol:
                for cand_out in self.sol:
                    delta_cost = self.obj_function.evaluate_exchange_cost(cand_in, cand_out, self.sol)
                    if delta_cost < min_delta_cost:
                        min_delta_cost = delta_cost
                        best_cand_in = cand_in
                        best_cand_out = cand_out

            # Implementa o melhor movimento se houver melhoria (min_delta_cost < 0)
            if min_delta_cost < -1e-10: # Usa um pequeno epsilon para comparação de ponto flutuante
                if best_cand_out is not None:
                    self.sol.remove(best_cand_out)
                if best_cand_in is not None:
                    self.sol.add(best_cand_in)
                
                # Atualiza o custo da solução e o estado de cobertura
                self.obj_function.evaluate(self.sol)
            else:
                # Nenhum movimento de melhoria encontrado, encerra a busca local
                break

        return self.sol

# --- Fim dos Ajustes ---

def main():
    """Função principal para testar o GRASP_QBF"""
    # Assume que as instâncias são geradas por generate_instances.txt
    # e colocadas em um diretório 'input' em relação ao local de execução.
    
    # Exemplo de uso: python seu_script_grasp.py input/instance-01.txt 0.1 100
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        alpha = float(sys.argv[3]) if len(sys.argv) > 2 else 0.2 # Alpha padrão para GRASP [14]
        iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    else:
        # Padrão para teste se nenhum argumento for fornecido
        print("Uso: python grasp_max_sc_qbf.py <nome_arquivo_instancia> [alpha] [iteracoes]")
        print("Usando instância e parâmetros de teste padrão.")
        # Cria um conteúdo de arquivo de instância dummy para MAX-SC-QBF para teste
        # Este conteúdo deve corresponder ao formato de generate_instances.txt e Atividade 1
        # Exemplo da Atividade 1, Seção 3.1 [15]
        default_qbf_content = """5
2 3 2 2 2
1 2
2 3 4
1 4
3 5
4 5
3 1 -2 0 3
2 -1 2 1
-1 2 -2
4 0
5"""
        filename = 'max_sc_qbf_test_instance.txt'
        with open(filename, 'w') as f:
            f.write(default_qbf_content)
        
        alpha = 0.2
        iterations = 100

    print(f"Executando GRASP para MAX-SC-QBF em {filename} com alpha={alpha}, iteracoes={iterations}")
    start_time = time.time()
    
    grasp = GRASP_QBF(alpha=alpha, iterations=iterations, filename=filename)
    best_sol = grasp.solve()
    
    end_time = time.time()

    print("\n" + "=" * 40)
    print("Resultados do GRASP para MAX-SC-QBF:")
    print("=" * 40)
    # Lembrar que best_sol.cost é -ObjVal porque QBF_Inverse minimiza.
    # Converte de volta para o valor objetivo real para exibição.
    if best_sol.cost == float('inf'):
        print("Nenhuma solução factível encontrada após todas as iterações.")
    else:
        print(f"Melhor Solução Encontrada (Valor do Objetivo): {-best_sol.cost:.4f}")
        # Converte os índices de volta para 1-indexado para exibição
        print(f"Subconjuntos selecionados: {[s + 1 for s in best_sol.elements]}") 
    print(f"Tempo de execução total: {end_time - start_time:.3f} segundos")

if __name__ == "__main__":
    main()
