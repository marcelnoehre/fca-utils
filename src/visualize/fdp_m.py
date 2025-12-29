import numpy as np

from itertools import combinations
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

class FDP_Additive_Features():
    
    def __init__(self,
        context: FormalContext
        ):
        self.context = context
        self.lattice = ConceptLattice.from_context(self.context)
        self.attributes = self.context.attribute_names
        self.attribute_map = {name: i for i, name in enumerate(self.attributes)}
        self.attribute_closures = {
            m: set(context.intention(context.extension([m])))
            for m in self.attributes
        }
        self.N = len(self.attributes)


