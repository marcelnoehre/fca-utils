import numpy as np

from itertools import combinations
from scipy.optimize import minimize
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

from src.fca_utils.lattice import *

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
        self.epsilon = 1e-6
        self.concepts = self.lattice.to_networkx().nodes
        self.intents = {
            concept: [val for _, val in intent_of_concept(self.lattice, concept)]
            for concept in self.concepts
        }
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

    def _get_concept_pos(self, concept, vectors):
        return sum([self.vectors[m] for m in self.intents[concept]], np.zeros(2))

    def dist_concept_to_edge(self, c, a, b):
        if np.array_equal(a, b):
            return np.linalg.norm(c - a)
        
        # projection of c to the edge (a,b)
        t = np.dot(c - a, b - a) / np.dot(b - a, b - a)
        
        # reduce to segment (clamping)
        t = np.clip(t, 0.0, 1.0)
        
        return np.linalg.norm(c - (a + t * (b - a)))

    def energy_rep(self, flat_vectors):
        vectors = flat_vectors.reshape(-1, 2)
        positions = [
            self._get_concept_pos(concept, vectors)
            for concept in self.concepts
        ]
        
        e_rep = 0.0
        for c, pos in enumerate(positions):
            for (a, b) in cover_relations(self.lattice):
                # edges without concept c
                if c == a or c == b:
                    continue

                # add distance of c to edge (a,b)
                dist = self.dist_concept_to_edge(pos, positions[a], positions[b])
                e_rep += 1.0 / (dist + self.epsilon)

        return e_rep
    
    def energy_att(self, flat_vectors):
        vectors = flat_vectors.reshape(-1, 2)
        positions = [
            self._get_concept_pos(concept, vectors)
            for concept in self.concepts
        ]

        e_att = 0.0
        for (a, b) in cover_relations(self.lattice):
            # |pos(b) - pos(a)|^2
            e_att += np.sum((positions[b] - positions[a])**2)

        return e_att