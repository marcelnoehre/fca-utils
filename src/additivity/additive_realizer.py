from z3 import *
from fcapy.lattice import ConceptLattice

from src.fca_utils.lattice import *
from src.linear_extension.sat_realizer import SatRealizer

class AdditiveRealizer:
    '''
    Compute an additive realizer for a given concept lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice for which to compute the additive realizer.
    '''
    def __init__(self, lattice: ConceptLattice):
        self.lattice = lattice
        self.G = self.lattice.to_networkx()
        self.relations = self.G.edges
        self.concepts = self.G.nodes
        self.top = len(self.concepts)-1
        self.objects = {v for _, v in extent_of_concept(lattice, 0)}
        self.features = {v for _, v in intent_of_concept(lattice, self.top)}
        self.incomparable_pairs = incomparability_graph(self.lattice).edges
        
        self.base_vectors = {}
        for concept in self.concepts:
            extent = {v for _, v in extent_of_concept(lattice, concept)}
            intent = {v for _, v in intent_of_concept(lattice, concept)}
            complement_intent = {f for f in self.features if f not in intent}
            self.base_vectors[concept] = extent.union(complement_intent)

        self._compute_dimension()
        self._setup_smt_variables()
        self._setup_relations()

    def _compute_dimension(self):
        self.solver = Solver()
        self.dimension, _ = SatRealizer(self.lattice).realizer()
        self.dimensions = [chr(97 + i) for i in range(self.dimension)]
        
    def _setup_smt_variables(self):
        self.smt_variables = {
            (d, v): Int(f'{d}_{v}')
            for d in self.dimensions
            for v in self.features.union(self.objects)
        }
        for d in self.dimensions:
            for concept in self.concepts:
                base_vectors = (self.smt_variables[d, var] for var in self.base_vectors[concept])
                self.solver.add(Int(f'{d}_{concept}') == sum(base_vectors))

    def _setup_relations(self):
        for a, b in self.relations:
            for d in self.dimensions:
                # if <= then the vector sum has to be >
                self.solver.add(Int(f'{d}_{a}') > Int(f'{d}_{b}'))

        for a, b in self.incomparable_pairs:
            # TODO: adjust for dimensions >= 2
            # (xa < xb and ya > yb) or (xa > xb and yb < ya)
            xa = Int(f'{self.dimensions[0]}_{a}')
            ya = Int(f'{self.dimensions[1]}_{a}')
            xb = Int(f'{self.dimensions[0]}_{b}')
            yb = Int(f'{self.dimensions[1]}_{b}')

            self.solver.add(Or(And(xa < xb, ya > yb), And(xa > xb, ya < yb)))
        
        for d in self.dimensions:
            self.solver.add(Int(f'{d}_0') == self.top)

    def realizer(self):
        if self.solver.check() == sat:
            self.model = self.solver.model()
            realizer = [
                [self.model[Int(f'{d}_{concept}')] for concept in self.concepts]
                for d in self.dimensions
            ]
            return self.dimension, realizer
        else:
            raise ValueError('No additive realizer found!')