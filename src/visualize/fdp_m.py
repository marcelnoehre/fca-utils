import numpy as np

from itertools import combinations
from scipy.optimize import minimize
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
        self._sup_inf_distance()
        self._optimize_init_energy()

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


    def _optimize_init_energy(self):
        optimized = list(self.attribute_map.values())

        def _energy_si(opt):
            total_energy = 0
            for i, j in combinations(range(self.N), 2):
                dsi = abs(self.dsi_matrix[i, j])
                if dsi > 0:
                    dist = abs(opt[i] - opt[j])
                    # (|pos(i) - pos(j)| - d_SI(i, j))^2
                    total_energy += (dsi - dist)**2

            return total_energy

        res = minimize(_energy_si, optimized, method='BFGS')
        self.vectors = {
            m: np.array([res.x[i], -1])
            for m, i in self.attribute_map.items()
        }