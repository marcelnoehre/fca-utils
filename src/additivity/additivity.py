import copy

from collections import deque
from fcapy.lattice import ConceptLattice
from typing import Iterable, Dict, Tuple, Set, List
from sympy import symbols, Eq, solve, sympify

from src.fca_utils.context import *
from src.fca_utils.lattice import *

class AdditivityCheck():

    def __init__(self,
            lattice: ConceptLattice,
            realizer: List[List[int]],
            coordinates: Dict[int, Tuple[int]]
        ):
        self.lattice = lattice
        self.dimension = len(realizer)
        self.realizer = tuple(
            realizer[i] if realizer[i][0] == 0 else list(reversed(realizer[i])) 
            for i in range(self.dimension)
        )
        self.coordinates = coordinates
        self.objects = extent_of_concept(self.lattice, self.realizer[0][0])
        self.features = intent_of_concept(self.lattice, self.realizer[0][-1])
        self.top = 0
        self.bottom = len(lattice.to_networkx().nodes) - 1
        self.dimensions = [chr(97 + i) for i in range(self.dimension)]

    def check_bottom_up_additivity(self):
        '''
        Check bottom-up additivity of the coordinate assignment.

        X := G
        (A, B) -> A
        
        Assign vectors to join-irreducible elements and sum them up the lattice

        Returns
        -------
        additive : bool
        '''
        join_irreducibles = {
            node: tuple(self.coordinates[node][i] - self.coordinates[list(self.lattice.children_dict[node])[0]][i]
                for i in range(len(self.realizer))
            )
            for node in get_join_irreducibles(self.lattice)
        }
        # root node (bottom) as (0, 0)
        # vector if join-irreducibles
        # sum of join-irreducibles for other nodes
        base_vectors_bottom = copy.deepcopy(join_irreducibles)
        base_vectors_bottom[self.bottom] = self.coordinates[self.bottom]

        self.bottom_up_additive = {}
        self.bottom_up_additive[self.bottom] = self.coordinates[self.bottom]

        queue = deque(self.lattice.parents(self.bottom))
        while queue:
            node = queue.popleft()
            children = all_children(self.lattice, node)
            
            # all children have to be processed first
            if all(child in self.bottom_up_additive for child in children):
                if node in join_irreducibles:
                    child = self.lattice.children(node)
                    child_vector = base_vectors_bottom[list(child)[0]]
                    
                    # if the child is a join-irreducible, sum up the chain until a non-join-irreducible is found
                    while list(child)[0] in join_irreducibles:
                        child = self.lattice.children(list(child)[0])
                        child_vector = tuple(child_vector[i] + base_vectors_bottom[list(child)[0]][i] for i in range(len(self.realizer)))

                    # base vector of join-irreducible + base vector of the single child
                    self.bottom_up_additive[node] = tuple(base_vectors_bottom[node][i] + child_vector[i] for i in range(len(self.realizer)))

                else:
                    pos = tuple(0 for _ in self.realizer)
                    for child in all_children(self.lattice, node):
                        # sum base vectors of all join-irreducible children
                        if child in join_irreducibles:
                            pos = tuple(pos[i] + base_vectors_bottom[child][i] for i in range(len(self.realizer)))
                    
                    # add sum as base vector for further nodes depending on this node
                    base_vectors_bottom[node] = pos
                    self.bottom_up_additive[node] = pos

                # add parents if not already processed or in queue
                for p in self.lattice.parents(node):
                    if p not in queue and p not in self.bottom_up_additive:
                        queue.append(p)

            else:
                # re-add to queue if children not processed yet
                queue.append(node)

        return self.bottom_up_additive == self.coordinates
    
    def check_top_down_additivity(self):
        '''
        Check top-down additivity of the coordinate assignment.

        X := M
        (A, B) -> M \\ B
        
        Assign vectors to meet-irreducible elements and sum them up the lattice

        Returns
        -------
        additive : bool
        '''
        meet_irreducibles = {
            node: tuple(self.coordinates[list(self.lattice.parents_dict[node])[0]][i] - self.coordinates[node][i] for i in range(len(self.realizer)))
            for node in get_meet_irreducibles(self.lattice)
        }
        # base vectors:
        # root node (top) as (0, 0)
        # vector if meet-irreducibles
        # sum of meet-irreducibles for other nodes
        base_vectors_top = copy.deepcopy(meet_irreducibles)
        base_vectors_top[self.top] = tuple(0 for _ in self.realizer)

        self.top_down_additive = {}
        self.top_down_additive[self.top] = self.coordinates[self.top]

        queue = deque(self.lattice.children(self.top))
        while queue:
            node = queue.popleft()
            parents = all_parents(self.lattice, node)
            
            # all parents have to be processed first
            if all(parent in self.top_down_additive for parent in parents):
                if node in meet_irreducibles:
                    parent = self.lattice.parents(node)
                    parent_vector = base_vectors_top[list(parent)[0]]
                    
                    # if the parent is a meet-irreducible, sum up the chain until a non-meet-irreducible is found
                    while list(parent)[0] in meet_irreducibles:
                        parent = self.lattice.parents(list(parent)[0])
                        parent_vector = tuple(parent_vector[i] + base_vectors_top[list(parent)[0]][i] for i in range(len(self.realizer)))

                    # top node - (base vector of meet-irreducible + base vector of the single parent)
                    # ensures a positive base vector from parent to child
                    self.top_down_additive[node] = tuple(self.coordinates[self.top][i] - (base_vectors_top[node][i] + parent_vector[i]) for i in range(len(self.realizer)))

                else:
                    pos = tuple(0 for _ in self.realizer)
                    for parent in parents:
                        if parent in meet_irreducibles:
                            # sum base vectors of all meet-irreducible parents
                            pos = tuple(pos[i] + base_vectors_top[parent][i] for i in range(len(self.realizer)))

                    # add sum as base vector for further nodes depending on this node
                    base_vectors_top[node] = pos
                    self.top_down_additive[node] = tuple(self.coordinates[self.top][i] - pos[i] for i in range(len(self.realizer)))
                    
                # add children if not already processed or in queue
                for p in self.lattice.children(node):
                    if p not in queue and p not in self.top_down_additive:
                        queue.append(p)

            else:
                # re-add to queue if parents not processed yet
                queue.append(node)

        return self.top_down_additive == self.coordinates
    
    def check_combined_additivity(self):
        '''
        Check combined additivity of the coordinate assignment.

        X := G U M
        (A, B) -> A U (M \\ B)

        Sum up the base vectors assigned to each concept by the objects and missing features.

        Returns
        -------
        additive : bool
        '''
        self.combined_equations = []
        visited = set()
        queue = deque([self.bottom])
        base_vectors_combined = {}

        while queue:
            node = queue.popleft()

            # objects and missing features for the current node
            extent = extent_of_concept(self.lattice, node)
            complement_intent = [f for f in self.features if f not in intent_of_concept(self.lattice, node)]
            base_vectors_combined[node] = (list(extent), complement_intent)
            vars = [v for _, v in extent] + [v for _, v in complement_intent]

            # construct equations for the current node
            for i, dim in enumerate(self.dimensions):
                self.combined_equations.append((' + '.join(f'{dim}_{v}' for v in vars) if vars else '0') + f' = {self.coordinates[node][i]}')

            # add parents if not already processed or in queue
            visited.add(node)
            for p in self.lattice.parents(node):
                if p not in queue and p not in visited:
                    queue.append(p)

        solver = LinearEquationSolver(
            lattice=self.lattice,
            coordinates=self.coordinates,
            base_vectors=base_vectors_combined,
            variables=[f'{dim}_{v}' for dim in self.dimensions for _, v in list(self.objects) + list(self.features)],
            equations=self.combined_equations,
            dimensions=self.dimensions
        )
        
        success, vector_variables = solver.solve_linear_equations()
        if not success:
            return False
        
        self.combined_additive = {}
        for node in self.lattice.to_networkx().nodes:
            pos = tuple(0 for _ in self.dimensions)
            for i, dim in enumerate(self.dimensions):
                for vec in [s for group in base_vectors_combined[node] for (_, s) in group]:

                    
                    pos = tuple((pos[dim] + vector_variables[f'{self.dimensions[dim]}_{vec}'] if i == dim else pos[dim]) for dim in range(len(self.dimensions)))
            self.combined_additive[node] = pos

        return self.combined_additive == self.coordinates
      
class LinearEquationSolver:
    '''
    Solve a system of linear equations derived from a concept lattice structure.

    Parameters
    ----------
    lattice : ConceptLattice
        The underlying concept lattice.
    coordinates : Dict[int, Tuple[int]]
        A mapping from lattice node indices to their coordinate tuples.
    base_vectors : Dict[int, Tuple[Iterable[Set[int]], Iterable[Set[int]]]]
        A mapping from lattice node indices to their base vectors.
    variables : Iterable[str]
        The variable names used in the equations.
    equations : Iterable[str]
        The linear equations to be solved, represented as strings.
    dimensions : Iterable[str]
        The dimensions for which the equations are defined.
    '''
    def __init__(self,
            lattice: ConceptLattice,
            coordinates: Dict[int, Tuple[int]],
            base_vectors: Dict[int, Tuple[Iterable[Set[int]], Iterable[Set[int]]]],
            variables: Iterable[str],
            equations: Iterable[str],
            dimensions: Iterable[str]
        ):
        self.lattice = lattice
        self.coordinates = coordinates
        self.base_vectors = base_vectors

        self.dimensions = dimensions
        self.variables = [f'{var}' for var in variables]
        self.symbols = symbols(' '.join(self.variables))
        self.equations = [Eq(sympify(l), sympify(r)) for l,r in (eq.split(' = ') for eq in equations)]
        self.solution = solve(self.equations, self.symbols, dict=True)

    def _solve(self, dim: str, node: int, expected: int):
        '''
        Solve the equation for a specific node and dimension.
        
        Parameters
        ----------
        dim : str
            The dimension for which to solve the equation.
        node : int
            The lattice node index.
        expected : int
            The expected value for the equation.
        '''
        eq = [] # construct equation to solve
        for var in [s for group in self.base_vectors[node] for (_, s) in group]:
            var = f'{dim}_{var}'

            # if variable value is already known, insert it directly
            if var in self.vector_variables:
                eq.append(str(self.vector_variables[var]))

            # if variable is free, insert symbol
            elif symbols(var) in self.free_vars:
                eq.append(str(var))

            else:
                # construct equation
                sub_eq = str(self.solution[0][symbols(var)]) 

                # insert values of already known variables
                for k, v in self.vector_variables.items():
                    sub_eq = sub_eq.replace(str(k), str(v))
                
                # solve if no variables are left in the sub-equation
                if not any(var in sub_eq for var in self.variables):
                    self.vector_variables[var] = int(eval(sub_eq, {"__builtins__": None}))
                    sub_eq = str(self.vector_variables[var])

                eq.append(f'({sub_eq})')

        # final equation to solve
        eq = ' + '.join(eq)

        # if the equation contains variables, solve it
        if any(var in eq for var in self.variables):
            # define sympy variables and equations
            expr_vars = [v for v in self.variables if eq.find(v) != -1]
            expr_symbols = symbols(' '.join(expr_vars)) 
            solution = solve(Eq(sympify(eq), expected), expr_symbols, dict=True)
            
            # if a solution is found, update the variable values
            if solution:
                for k, v in solution[0].items():
                    self.vector_variables[str(k)] = int(v)
            
            # if no solution is found the variables cancel out -> assign 0 to all variables
            else:
                for var in expr_vars:
                    if str(var) not in list(self.vector_variables.keys()):
                        self.vector_variables[str(var)] = 0

    def solve_linear_equations(self):
        '''
        Solve the system of linear equations.

        Returns
        -------
        success : bool
            True if the equations were successfully solved, False otherwise.
        vector_variables : Dict[str, int]
            A mapping from variable names to their solved integer values.
        '''
        if not self.solution:
            return False, None
        
        # identify free variables that can take any value
        self.free_vars = [v for v in self.symbols if v not in self.solution[0]]

        # extract already fixed variable values
        self.vector_variables = {}
        for k, v in self.solution[0].items():
            try:
                self.vector_variables[str(k)] = int(v)
            except TypeError:
                continue

        visited = set()
        queue = deque([len(self.lattice.to_networkx().nodes) - 1])

        while queue:
            node = queue.popleft()

            # solve dimensions separately
            for i, dim in enumerate(self.dimensions):
                while not all(f'{dim}_{var}' in self.vector_variables.keys() 
                    for var in [s for group in self.base_vectors[node] for (_, s) in group]
                ):
                    self._solve(dim, node, self.coordinates[node][i])
            
            # add parents if not already processed or in queue
            visited.add(node)
            for p in self.lattice.parents(node):
                if p not in queue and p not in visited:
                    queue.append(p)

        return True, self.vector_variables
