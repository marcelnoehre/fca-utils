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

    def _sup_inf_distance(self):
        self.dsi_matrix = np.zeros((self.N, self.N))
        for i, j in combinations(self.attributes, 2):
            m_i, m_j = self.attribute_map[i], self.attribute_map[j]
            intent_i = self.attribute_closures[i]
            intent_j = self.attribute_closures[j]

            # check if comparable
            if not (intent_i.issubset(intent_j) or intent_j.issubset(intent_i)):
                # inf_size - sup_size - 1 
                dsi = len(intent_i.intersection(intent_j)) - len(intent_i.union(intent_j)) - 1
                self.dsi_matrix[m_i][m_j] = dsi
                self.dsi_matrix[m_j][m_i] = dsi

