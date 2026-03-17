from z3 import *
from fcapy.context import FormalContext

from fca.context import Context
from fca.lattice import Lattice

class AdditiveRealizer:
    '''
    Compute an additive realizer for a given concept lattice.

    Parameters
    ----------
    context : FormalContext
        The formal context for which to compute the realizer.
    dimension : int
        The order dimension of the lattice
    '''
    def __init__(self, context: FormalContext, dimension: int):
        self.context = Context(context)
        self.lattice = Lattice(context)
        self.graph = self.lattice.lattice.to_networkx()
        self.relations = self.graph.edges
        self.concepts = self.graph.nodes
        self.top = len(self.concepts)-1
        
        self.objects = {v for _, v in self.lattice.extent_of_concept(0)}
        self.attributes = {v for _, v in self.lattice.intent_of_concept(self.top)}
        self.incomparable_pairs = self.lattice.incomparability_graph().edges
        
        self.base_vectors = {}
        for concept in self.concepts:
            extent = {v for _, v in self.lattice.extent_of_concept(concept)}
            intent = {v for _, v in self.lattice.intent_of_concept(concept)}
            complement_intent = {f for f in self.attributes if f not in intent}
            self.base_vectors[concept] = extent.union(complement_intent)

        self.solver = Solver()
        self.dimension = dimension
        self.dimensions = [chr(97 + i) for i in range(self.dimension)]

        self._setup_smt_variables()
        self._setup_relations()

    def realizer(self):
        '''
        Compute an additive realizer using the z3 SMT solver.

        Raises
        ------
        error : ValueError
             If no additive realizer is found for the concept lattice.
        '''
        if self.solver.check() == sat:
            # solved clauses

            self.model = self.solver.model()
            # prepare empty realizer
            realizer = {
                d: [None for _ in self.concepts]
                for d in self.dimensions
            }

            # derive base vectors
            for g in self.objects:
                self.base_vectors[g] = [
                    float(self.model[Real(f'{dim}_{g}')].as_fraction())
                    for dim in self.dimensions
                ]
            for m in self.attributes:
                self.base_vectors[m] = [
                    float(-self.model[Real(f'{dim}_{m}')].as_fraction())
                    for dim in self.dimensions
                ]

            # insert concepts based on their vector sum
            for d in self.dimensions:
                for concept in self.concepts:
                    realizer[d][self.model[Int(f'{d}_{concept}')].as_long()] = concept

            return self.dimension, [list(reversed(le)) for le in realizer.values()]
        else:
            raise ValueError('No additive realizer found!')
        
    def _setup_smt_variables(self):
        '''
        Define SMT variables for all concepts and base vectors.
        '''
        # base vectors
        self.smt_variables = {
            (d, v): Real(f'{d}_{v}')
            for d in self.dimensions
            for v in self.attributes.union(self.objects)
        }
        # concept = sum of base vectors
        # (A, B) -> A U (M \ B)
        for d in self.dimensions:
            for concept in self.concepts:
                base_vectors = (self.smt_variables[d, var] for var in self.base_vectors[concept])
                self.solver.add(Int(f'{d}_{concept}') == sum(base_vectors))

    def _setup_relations(self):
        '''
        Define SMT clauses for additivity.
        '''
        # related pairs: a < b
        for a, b in self.relations:
            for d in self.dimensions:
                # if <= then the vector sum has to be >
                self.solver.add(Int(f'{d}_{a}') > Int(f'{d}_{b}'))

        # incomparable pairs
        for a, b in self.incomparable_pairs:
            a_vars = [Int(f'{d}_{a}') for d in self.dimensions]
            b_vars = [Int(f'{d}_{b}') for d in self.dimensions]
            # at least one extension has a < b
            a_lt_b = [a_vars[i] < b_vars[i] for i in range(self.dimension)]
            # at least one extension has a > b
            a_gt_b = [a_vars[i] > b_vars[i] for i in range(self.dimension)]
            self.solver.add(And(Or(*a_lt_b), Or(*a_gt_b)))
            # a != b in the same dimension
            for d in self.dimensions:
                self.solver.add(Int(f'{d}_{a}') != Int(f'{d}_{b}'))

        # Fix bottom and top to define range
        for d in self.dimensions:
            self.solver.add(Int(f'{d}_{len(self.concepts)-1}') == 0)
            self.solver.add(Int(f'{d}_0') == self.top)
