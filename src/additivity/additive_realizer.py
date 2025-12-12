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

    def _define_sat_variables(self, dim: int):
        self.dimensions = [chr(97 + i) for i in range(dim)]
        self.sat_variables = {
            f'{d}_{bv}': i+1
            for i, (d, bv) in enumerate(
                (d, bv) for d in self.dimensions for bv in self.base_vectors
            )
        }
