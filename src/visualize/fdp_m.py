import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.optimize import minimize
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from scipy.spatial.distance import pdist, squareform

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
        self._initialize_vectors()
        self._optimize_layout()

    def _sup_inf_distance(self):
        self.dsi_matrix = np.zeros((self.N, self.N))
        for i, j in combinations(range(self.N), 2):
            d = len(self.attribute_closures[self.attributes[i]] ^ self.attribute_closures[self.attributes[j]])
            self.dsi_matrix[i, j] = self.dsi_matrix[j, i] = d

    def _solve_spring_model(self):
        def e_si(flat_pts):
            pts = flat_pts.reshape(-1, 2)
            e_si = 0
            for i, j in combinations(range(self.N), 2):
                dist = np.linalg.norm(pts[i] - pts[j])
                e_si += (dist - self.dsi_matrix[i, j])**2
            return e_si

        res = minimize(e_si, np.random.rand(self.N * 2), method='BFGS')
        return res.x.reshape(-1, 2)

    def _get_longest_path(self, spring_pts):
        # spring_pts is an (N, 2) array from the spring model
        dists = squareform(pdist(spring_pts))
        # Find indices of the maximum distance
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        return i, j, spring_pts[i], spring_pts[j]

    def _rotate_layout(self, spring_pts):
        i, j, p1, p2 = self._get_longest_path(spring_pts)
        
        # Vector of the longest path
        vec = p2 - p1
        angle = -np.arctan2(vec[1], vec[0])
        
        # Rotation matrix
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        
        # Rotate all points
        rotated_pts = spring_pts @ rot.T
        return rotated_pts
    
    def _initialize_vectors(self):
        spring_pts = self._solve_spring_model()
        rotated_pts = self._rotate_layout(spring_pts)
        order = np.argsort(rotated_pts[:, 0])
        
        self.vectors = {}
        for i, idx in enumerate(order):
            x = -1 + 2 * (i / (self.N - 1))
            self.vectors[self.attributes[idx]] = np.array([x, x**2]) + np.random.normal(0, 0.01, size=2)

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
        # Correct integration constants from thesis page 34
        E0 = -phi_0 - np.sin(phi_0) * np.cos(phi_0)
        E1 = (np.pi - phi_0) - np.sin(phi_0) * np.cos(phi_0)

        e_grav = 0.0

        for n_i in vectors:
            x, y = n_i
            phi = np.arctan2(y, x) # Returns (-pi, pi]

            # 1. Total Barrier for the lower half-plane (y <= 0)
            # If the vector points down or flat, we apply a massive penalty 
            # to force the optimizer back into the (0, pi) range.
            if phi <= 0:
                # We use a large constant plus a distance penalty
                e_grav += 1e6 * (abs(phi) + 1.0)
                continue

            # 2. Case 1: Angle is too small (too far right, phi < phi_0)
            if phi < phi_0:
                # The cotangent (cos/sin) correctly goes to infinity as phi -> 0
                e_grav += phi + (np.cos(phi) / np.sin(phi)) * (np.sin(phi_0)**2) + E0

            # 3. Case 2: Angle is too large (too far left, phi > pi - phi_0)
            elif phi > (np.pi - phi_0):
                # The cotangent correctly goes to infinity as phi -> pi
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