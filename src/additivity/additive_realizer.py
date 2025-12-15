import itertools

from sage.sat.solvers.satsolver import SAT

from fcapy.lattice import ConceptLattice
from sympy import symbols, Eq, solve, sympify

from src.fca_utils.lattice import *

class AdditiveRealizer:
    '''
    Compute an additive realizer for a given concept lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice for which to compute the additive realizer.
    '''

    def __init__(self, lattice: ConceptLattice):
        # store lattice and poset
        self.lattice = lattice
        self.concepts = self.lattice.to_networkx().nodes

        self.objects = {v for _, v in extent_of_concept(lattice, 0)}
        self.features = {v for _, v in intent_of_concept(lattice, len(self.concepts)-1)}
        self.base_vectors = self.features.union(self.objects)
        # store incomparable pairs
        self.incomparable_pairs = incomparability_graph(self.lattice)
        self.N_incomparable = len(self.incomparable_pairs)
        self.sat_variables = {}

    def realizer(self):
        dim = 2
        self._construct_linear_equations(dim)
        self._construct_additivity_clauses()

    def _construct_linear_equations(self, dim: int):
        self.dimensions = [chr(97 + i) for i in range(dim)]
        for d, bot in itertools.product(self.dimensions, [len(self.concepts)-1]):
            self.sat_variables[f'{d}_{bot}'] = len(self.sat_variables)+1

        self.variables = {
            f"{d}_{v}"
            for d in self.dimensions
            for v in self.features | self.objects
        } | {
            f"{d}_{n}"
            for d in self.dimensions
            for n in self.concepts
        }

        self.equations = []
        visited = set()
        queue = deque([len(self.concepts)-1])

        while queue:
            node = queue.popleft()

            # objects and missing features for the current node
            extent = [v for _, v in extent_of_concept(self.lattice, node)]
            intent = [v for _, v in intent_of_concept(self.lattice, node)]
            complement_intent = [f for f in self.features if f not in intent]
            variables = extent + complement_intent

            # construct equations for the current node
            for d in self.dimensions:
                self.sat_variables[f'{d}_{node}'] = len(self.sat_variables)+1
                self.equations.append((tuple(f'{d}_{v}' for v in variables) , f'{d}_{node}'))

            # add parents if not already processed or in queue
            visited.add(node)
            for p in self.lattice.parents(node):
                if p not in queue and p not in visited:
                    queue.append(p)

        self.symbols = symbols(self.variables)
        self.sympy_equations = [
            Eq(sympify(' + '.join(eq[0]) if eq[0] else '0'), sympify(eq[1]))
            for eq in self.equations
        ]
        self.solution = solve(self.sympy_equations, self.symbols, dict=True)

    def _construct_additivity_clauses(self):
        for d, bv in itertools.product(self.dimensions, self.base_vectors):
            self.sat_variables[f'{d}_{bv}'] = len(self.sat_variables)+1
        # A -> B = ¬A v B
        self.additivity_clauses = []
        for target, vectors in self.solution[0].items():
            vectors = str(vectors).split(' + ')
            if not vectors == ['0']:
                self.additivity_clauses.append(
                    [-self.sat_variables[v] for v in vectors]
                    + [self.sat_variables[str(target)]]
                )