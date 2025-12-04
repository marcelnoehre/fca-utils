from collections import deque
from fcapy.lattice import ConceptLattice
from typing import Iterable, Dict, Tuple, Set
from sympy import symbols, Eq, solve, sympify

from src.fca_utils.context import *
from src.fca_utils.lattice import *

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

        self.variables = [f'{dim}_{var}' for dim in self.dimensions for var in variables]
        self.symbols = symbols(' '.join(self.variables))
        self.equations = [Eq(sympify(l), sympify(r)) for l,r in (eq.split('=') for eq in equations)]
        self.dimensions = dimensions

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

        return self.vector_variables
