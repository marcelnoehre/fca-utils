import itertools

from sage.sat.solvers.satsolver import SAT

from fcapy.lattice import ConceptLattice

from src.fca_utils.lattice import *
from src.fca_utils.parser import sage_poset_from_lattice

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
        self.P = sage_poset_from_lattice(lattice)
        self.concepts = self.P.list()

        self.objects = {v for _, v in extent_of_concept(lattice, 0)}
        self.features = {v for _, v in intent_of_concept(lattice, len(self.concepts)-1)}
        self.base_vectors = self.features.union(self.objects)
        # store incomparable pairs
        self.incomparable_pairs = self.P.incomparability_graph().edges(sort=True, labels=False)
        self.N_incomparable = len(self.incomparable_pairs)

    def realizer(self):
        dim = 2
        self._define_sat_variables(dim)
        self._construct_linear_equations()
        self._construct_additivity_clauses()

    def _define_sat_variables(self, dim: int):
        self.dimensions = [chr(97 + i) for i in range(dim)]
        self.sat_variables = {
            f'{d}_{bv}': i+1
            for i, (d, bv) in enumerate(itertools.product(self.dimensions, self.base_vectors))
        }

    def _construct_linear_equations(self):
        self.equations = []
        visited = set()
        queue = deque([node for node in self.lattice.parents(len(self.concepts)-1)])

        while queue:
            node = queue.popleft()

            # objects and missing features for the current node
            extent = [v for _, v in extent_of_concept(self.lattice, node)]
            complement_intent = [f for f in self.features if f not in intent_of_concept(self.lattice, node)]
            vars = extent + complement_intent

            # construct equations for the current node
            for d in self.dimensions:
                self.sat_variables[f'{d}_{node}'] = len(self.sat_variables)
                self.equations.append((tuple(f'{d}_{v}' for v in vars) if vars else tuple('0'), f'{d}_{node}'))

            # add parents if not already processed or in queue
            visited.add(node)
            for p in self.lattice.parents(node):
                if p not in queue and p not in visited:
                    queue.append(p)

    def _construct_additivity_clauses(self):
        # A -> B = ¬A v B
        self.additivity_clauses = [
            ([-self.sat_variables[v] for v in eq] + [self.sat_variables[y]])
            for eq, y in self.equations
        ]