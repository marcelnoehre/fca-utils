import numpy as np
import matplotlib.pyplot as plt

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
        self._optimize_layout()

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
            m: np.array([res.x[i], 1.0])
            for m, i in self.attribute_map.items()
        }

    def _get_concept_pos(self, concept, vectors):
        indices = [self.attribute_map[m] for m in self.intents[concept]]
        if not indices:
            return np.zeros(2)
        return np.sum(vectors[indices], axis=0)

    def _optimize_layout(self):
        res = minimize(
            self._total_energy,
            np.array([self.vectors[m] for m in self.attributes]).flatten(),
            method='CG',
            options={'maxiter': 100, 'disp': True}
        )
        optimized_flat = res.x
        optimized_matrix = optimized_flat.reshape(-1, 2)
        for i, m in enumerate(self.attributes):
            print('old', self.vectors[m])
            print('new', optimized_matrix[i])
            self.vectors[m] = optimized_matrix[i]
    
    def _total_energy(self, flat_vectors):
        # forces
        e_rep = self._energy_rep(flat_vectors)
        e_att = self._energy_att(flat_vectors)
        e_grav = self._energy_grav(flat_vectors)

        # weights
        r = 1.0
        a = 0.5
        g = 0.2

        return r * e_rep + a * e_att + g * e_grav

    def _energy_rep(self, flat_vectors):
        vectors = flat_vectors.reshape(-1, 2)
        positions = [
            self._get_concept_pos(concept, vectors)
            for concept in self.concepts
        ]

        def _dist_concept_to_edge(c, a, b):
            if np.array_equal(a, b):
                return np.linalg.norm(c - a)
            
            # projection of c to the edge (a,b)
            t = np.dot(c - a, b - a) / np.dot(b - a, b - a)
            
            # reduce to segment (clamping)
            t = np.clip(t, 0.0, 1.0)
            
            return np.linalg.norm(c - (a + t * (b - a)))
        
        e_rep = 0.0
        for c, pos in enumerate(positions):
            for (a, b) in cover_relations(self.lattice):
                # edges without concept c
                if c == a or c == b:
                    continue

                # add distance of c to edge (a,b)
                dist = _dist_concept_to_edge(pos, positions[a], positions[b])
                e_rep += 1.0 / (dist + self.epsilon)

        return e_rep
    
    def _energy_att(self, flat_vectors):
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
    
    def _energy_grav(self, flat_vectors):
        vectors = flat_vectors.reshape(-1, 2)
        
        phi_0 = np.pi / (self.N + 1)
        E0 = -(phi_0 + np.sin(phi_0) * np.cos(phi_0))
        E1 = (np.pi - phi_0) - np.sin(phi_0) * np.cos(phi_0)

        e_grav = 0.0

        for n_i in vectors:
            x, y = n_i

            # (0, pi)
            phi = np.arctan2(y, x)

            # Case 1: too far on right
            if 0 < phi < phi_0:
                e_grav += phi + (np.cos(phi) / np.sin(phi)) * (np.sin(phi_0)**2) + E0

            # Case 2: too far on left
            elif (np.pi - phi_0) < phi < np.pi:
                e_grav += -phi - (np.cos(phi) / np.sin(phi)) * (np.sin(phi_0)**2) + E1

        return e_grav
    
    def plot(self):
        plt.figure(figsize=(8, 6))

        coordinates = {}
        for concept in self.concepts:
            x, y = np.array(sum([self.vectors[m] for m in self.intents[concept]], np.zeros(2)))
            coordinates[concept] = (x, -y)
            plt.scatter(x, -y, color="blue", zorder=3)
            plt.text(x, -y - 0.075 * self.N, concept, fontsize=12, ha='center', va='bottom', color='grey')

        for (a, b) in cover_relations(self.lattice):
            x0, y0 = np.array(coordinates[a])
            x1, y1 = np.array(coordinates[b])
            plt.plot([x0, x1], [y0, y1], color="black", zorder=2)

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.show()