import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.optimize import minimize
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from scipy.spatial.distance import pdist, squareform

from src.fca_utils.lattice import *

class FDP_Additive_Centralized():
    '''
    Force Directed Placement Algorithm to compute a centralized additive drawing of a
    Concept Lattice.

    Parameters
    ----------
    context : FormalContext
        The Formal Context
    repetitions : int
        Repetitions with different initial spring layouts

    Reference
    ---------
    Zschalig, Christian. Ein Force Directed Placement Algorithmus zum Zeichnen von
    Liniendiagrammen von Verbände, 2002.
    '''
    def __init__(self,
            context: FormalContext,
            repetitions: int
        ):
        self.context = context
        self.lattice = ConceptLattice.from_context(self.context)
        # attributes
        self.attributes = self.context.attribute_names
        self.attribute_map = {name: i for i, name in enumerate(self.attributes)}
        self.attribute_closures = {
            m: set(context.intention(context.extension([m])))
            for m in self.attributes
        }
        self.N_m = len(self.attributes)
        # objects
        self.objects = self.context.object_names
        self.object_map = {name: i + len(self.attributes) for i, name in enumerate(self.objects)}
        self.object_closures = {
            g: set(context.extension(context.intention([g])))
            for g in self.objects
        }
        self.object_descriptions = {
            g: set(context.intention([g]))
            for g in self.objects
        }
        self.N_g = len(self.objects)
        # general
        self.N = self.N_m + self.N_g
        self.concepts = self.lattice.to_networkx().nodes
        self.intents = {
            concept: [val for _, val in intent_of_concept(self.lattice, concept)]
            for concept in self.concepts
        }
        self.extents = {
            concept: [val for _, val in extent_of_concept(self.lattice, concept)]
            for concept in self.concepts
        }
        # FDP
        self.score = np.inf
        self.epsilon = 1e-6
        self._sup_inf_distance()
        for _ in range(repetitions):
            self._initialize_vectors()
            self._optimize_layout()

    def plot(self):
        '''
        Plot the Concept Lattice.
        '''
        plt.figure(figsize=(8, 6))

        # vertices
        for concept in self.concepts:
            (x, y) = self.coordinates[concept]
            plt.scatter(x, y, color="blue", zorder=3)
            plt.text(x, y - 0.075, concept, fontsize=12, ha='center', va='bottom', color='grey')

        # edges
        for (a, b) in cover_relations(self.lattice):
            x0, y0 = np.array(self.coordinates[a])
            x1, y1 = np.array(self.coordinates[b])
            plt.plot([x0, x1], [y0, y1], color="black", zorder=2)

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _sup_inf_distance(self):
        '''
        Supremum Infimum Distance for each combination of attributes.
        '''
        self.dsi_matrix = np.zeros((self.N_m + self.N_g, self.N_m + self.N_g))
        for i, j in combinations(range(self.N_m + self.N_g), 2):
            # M x M
            if i < self.N_m and j < self.N_m:
                d = len(self.attribute_closures[self.attributes[i]] ^
                        self.attribute_closures[self.attributes[j]])
            # G x G
            elif i >= self.N_m and j >= self.N_m:
                d = len(self.object_closures[self.objects[i - self.N_m]] ^
                        self.object_closures[self.objects[j - self.N_m]])
            # (M x G) v (G x M)
            else:
                # attributes of i
                if i < self.N_m:
                    val_i = self.attribute_closures[self.attributes[i]]
                else:
                    val_i = self.object_descriptions[self.objects[i - self.N_m]]
                # attributes of j
                if j < self.N_m:
                    val_j = self.attribute_closures[self.attributes[j]]
                else:
                    val_j = self.object_descriptions[self.objects[j - self.N_m]]

                d = len(val_i ^ val_j)

            self.dsi_matrix[i, j] = self.dsi_matrix[j, i] = d

    def _solve_spring_model(self):
        '''
        Compute initial spring model by minimizing the systems energy.
        '''
        def e_si(flat_pts):
            '''
            Compute the energy of the sup-inf system. 
            '''
            pts = flat_pts.reshape(-1, 2)
            e_si = 0
            for i, j in combinations(range(self.N_m + self.N_g), 2):
                # (|n_i - n_j| - d_SI(n_i, n_j))^2
                dist = np.linalg.norm(pts[i] - pts[j])
                e_si += (dist - self.dsi_matrix[i, j])**2
            return e_si

        res = minimize(e_si, np.random.rand((self.N_m + self.N_g) * 2), method='BFGS')
        return res.x.reshape(-1, 2)

    def _initialize_vectors(self):
        '''
        Compute initial object and attribute vectors by minimizing the difference
        between the geometric distance and sup inf distance.
        '''
        # intial spring layout
        spring_pts = self._solve_spring_model()
        # center around origin
        spring_pts -= np.mean(spring_pts, axis=0)
        spring_pts *= 0.5
        
        self.vectors = {}
        for i in range(self.N):
            vec = spring_pts[i].copy()
            if i < self.N_m: # Attribute nach UNTEN
                if vec[1] >= 0: vec[1] = -abs(vec[1]) - 0.1 # Erzwinge negatives y
            else: # Gegenstände nach OBEN
                if vec[1] <= 0: vec[1] = abs(vec[1]) + 0.1  # Erzwinge positives y
            
            # In dict speichern (Index-basiert für den Optimierer)
            if i < self.N_m:
                self.vectors[self.attributes[i]] = vec
            else:
                self.vectors[self.objects[i - self.N_m]] = vec

    def _get_concept_pos(self, concept, vectors):
        '''
        Compute concept position based on actual vectors.

        Parameters
        ----------
        concept : int
            Concept to compute the position for
        vectors : Dict[int, np.array]
            Dictionary assigning vectors to attributes  

        Returns
        -------
        position : np.array
            Position of the concept
        '''
        indices_m = [self.attribute_map[m] for m in self.intents[concept]]
        indices_g = [self.object_map[g] for g in self.extents[concept]]

        # (0, 0) if concept has no objects and no attributes
        if not indices_m and not indices_g:
            return np.zeros(2)
        
        sum_g = np.sum(vectors[indices_g], axis=0) if indices_g else np.zeros(2)
        sum_m = np.sum(vectors[indices_m], axis=0) if indices_m else np.zeros(2)
        
        return sum_g + sum_m
    
    def _optimize_layout(self):
        '''
        Compute an optimized layout based on the initial spring layout.
        '''
        res = minimize(
            self._total_energy,
            np.array([self.vectors[m] for m in self.attributes] + [self.vectors[g] for g in self.objects]).flatten(),
            method='CG',
            options={'maxiter': 1000, 'disp': True}
        )
        optimized_matrix = res.x.reshape(-1, 2)
        for i, m in enumerate(self.attributes):
            self.vectors[m] = optimized_matrix[i]
        for i, g in enumerate(self.objects):
            self.vectors[g] = optimized_matrix[i + self.N_m]
        
        if res.fun < self.score:
            self.coordinates = {}
            for concept in self.concepts:
                m_vectors = [self.vectors[m] for m in self.intents[concept]]
                g_vectors = [self.vectors[g] for g in self.extents[concept]]
                x, y = np.array(sum(m_vectors + g_vectors))
                self.coordinates[concept] = (x, y)

    def _total_energy(self, flat_vectors):
        '''
        Compute the total energy of forces in the drawing of the Concept Lattice.
        1. Repulsive Energy (E_rep):
            Maximizes distance between nodes and non-incident edges.
        2. Attractive Energy (E_att):
            Minimizes edge lengths to keep related concepts close.
        3. Gravitational Energy (E_grav): 
            Constraints attribute vectors to "safe" angles to ensure an upward-directed, readable diagram.

        Parameters
        ----------
        flat_vectors : np.array
            A 1D flattened array of attribute vectors.

        Returns
        -------
        energy : float
            The total energy of the three forces.
        '''
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
        '''
        Compute the Repulsive Energy, which maximizes the distance between nodes and non-incident edges.

        Parameters
        ----------
        flat_vectors : np.array
            A 1D flattened array of attribute vectors.

        Returns
        -------
        energy : float
            The repulsive energy.
        '''
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
        '''
        Compute the Attractive Energy, which minimizes edge lengths to keep related concepts close.

        Parameters
        ----------
        flat_vectors : np.array
            A 1D flattened array of attribute vectors.

        Returns
        -------
        energy : float
            The attractive energy.
        '''
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
        '''
        Compute the Gravitational Energy for a centralized additive model.
        Attribute vectors (m) are constrained to the lower half-plane (y < 0).
        Object vectors (g) are constrained to the upper half-plane (y > 0).
        '''
        vectors = flat_vectors.reshape(-1, 2)
        
        vectors_m = vectors[:self.N_m]
        vectors_g = vectors[self.N_m:]
        
        phi_0 = np.pi / (self.N_m + self.N_g + 1)
        E0 = -phi_0 - np.sin(phi_0) * np.cos(phi_0)
        E1 = (np.pi - phi_0) - np.sin(phi_0) * np.cos(phi_0)

        e_grav = 0.0

        # gravitation for object vectors (upper half, y > 0)
        for n_i in vectors_g:
            x, y = n_i
            phi = np.arctan2(y, x)  # (-pi, pi]

            if phi <= 0:
                e_grav += 1e6 * (abs(phi) + 1.0)
                continue

            if phi < phi_0:
                e_grav += phi + (np.cos(phi) / np.sin(phi)) * (np.sin(phi_0)**2) + E0
            elif phi > (np.pi - phi_0):
                e_grav += -phi - (np.cos(phi) / np.sin(phi)) * (np.sin(phi_0)**2) + E1

        # gravitation for attribute vectors (upper half, y > 0)
        for n_i in vectors_m:
            x, y = n_i
            phi = np.arctan2(-y, x) 

            if phi <= 0:
                e_grav += 1e6 * (abs(phi) + 1.0)
                continue

            if phi < phi_0:
                e_grav += phi + (np.cos(phi) / np.sin(phi)) * (np.sin(phi_0)**2) + E0
            elif phi > (np.pi - phi_0):
                e_grav += -phi - (np.cos(phi) / np.sin(phi)) * (np.sin(phi_0)**2) + E1

        return e_grav