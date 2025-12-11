from sage.all import Poset
from sage.sat.solvers.satsolver import SAT

from typing import List, Dict
from fcapy.lattice import ConceptLattice

from src.fca_utils.parser import sage_poset_from_lattice


class AdditiveRealizer:
    '''
    Compute a additive realizer for a given concept lattice.

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
        # store incomparable pairs
        self.incomparable_pairs = self.P.incomparability_graph().edges(sort=True, labels=False)
        self.N_incomparable = len(self.incomparable_pairs)
        # assign SAT variables to incomparable pairs
        # positive for (a,b), negative for (b,a)
        self.sat_variables = {}
        for i, (a, b) in enumerate(self.incomparable_pairs, start=1):
            self.sat_variables[(a, b)] = i
            self.sat_variables[(b, a)] = -i

    def normal_realizer(self):
        '''
        Compute an realizer independent of additvity constraints using SAT solver.

        Returns:
            dimension : int
                Order dimension of the poset.
            realizer : List[Poset]
                List of linear extensions forming the realizer.
        '''
        # setup clauses for SAT solver
        self._setup_incomparability_clauses()

        dim = 1
        while True:
            result = self._build_sat(dim)()
            if result is not False:
                break
            dim += 1
        
        return dim, self._realizer_from_sat_result(result, dim)

    def _setup_incomparability_clauses(self):
        '''
        Setup clauses to ensure transitivity among incomparable pairs.
        For each triple (a, b, c), ensure that if a < b and b < c, then a < c.
        '''
        self.incomparability_clauses = []
        for (a, c), ac_var in self.sat_variables.items():
            for b in self.concepts:
                # avoid trivial transitivity checks
                if b == a or b == c:
                    continue

                # get SAT variables for pairs
                ab_var = self.sat_variables.get((a, b))
                bc_var = self.sat_variables.get((b, c))

                # skip if (a,b) or (b,c) is not relevant for transitivity
                if ab_var is None and not self.P.is_less_than(a, b):
                    continue
                if bc_var is None and not self.P.is_less_than(b, c):
                    continue

                # Build clause: (~(a<b) OR ~(b<c) OR (a<c))
                clause = []
                if ab_var is not None:
                    clause.append(-ab_var)
                if bc_var is not None:
                    clause.append(-bc_var)
                clause.append(ac_var)

                self.incomparability_clauses.append(clause)

    def _build_sat(self, dim: int) -> SAT:
        '''
        Build the SAT solver instance for given dimension.

        Parameters
        ----------
        dim : int
            Dimension of the poset (number of linear extensions).
        
        Returns
        -------
        sat : SAT
            SAT solver instance with added clauses.
        '''
        sat = SAT(solver="kissat")
        
        # Incomparability clauses for transitivity
        sign = lambda x: 1 if x > 0 else -1

        # Ensure independent variables for each linear extension
        for offset in [i * self.N_incomparable for i in range(dim)]:
            for clause in self.incomparability_clauses:
                sat.add_clause([var + sign(var) * offset for var in clause])

        for var in range(1, self.N_incomparable + 1):
            # positive clause: at least one extension has a < b
            sat.add_clause([var + i * self.N_incomparable for i in range(dim)])
            # negative clause: at least one extension has b < a
            sat.add_clause([-var - i * self.N_incomparable for i in range(dim)])

        return sat
    
    def _realizer_from_sat_result(self, result: Dict[int, bool], dim: int) -> List[Poset]:
        '''
        Construct the realizer from the SAT solver result.

        Parameters
        ----------
        result : Dict[int, bool]
            Mapping from SAT variable to boolean value indicating the order.
        dim : int
            Dimension of the poset (number of linear extensions).

        Returns
        -------
        realizer : List[Poset]
            List of linear extensions forming the realizer.
        '''
        realizer = []

        for i in range(dim):
            linear_extension = self.P.hasse_diagram().copy()

            # offset for i-th linear extension
            offset = i * self.N_incomparable

            # reach total order by adding edges between incomparable pairs according to SAT result
            for var_i, (a, b) in enumerate(self.incomparable_pairs, start=1):
                if result[var_i + offset]:
                    linear_extension.add_edge(a, b)
                else:
                    linear_extension.add_edge(b, a)

            # topological sort gives the linear extension
            realizer.append(linear_extension.topological_sort())

        return realizer
