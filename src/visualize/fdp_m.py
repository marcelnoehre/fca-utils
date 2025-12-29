import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.optimize import minimize
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from scipy.spatial.distance import pdist, squareform

from src.fca_utils.lattice import *

class FDP_Additive_Features():
    '''
    Force Directed Placement Algorithm to compute an attribute additive drawing of a
    Concept Lattice.

    Parameters
    ----------
    context : FormalContext
        The Formal Context

    Reference
    ---------
    Zschalig, Christian. Ein Force Directed Placement Algorithmus zum Zeichnen von
    Liniendiagrammen von Verbände, 2002.
    '''
    def __init__(self, context: FormalContext):
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

    def plot(self):
        '''
        Plot the Concept Lattice.
        '''
        plt.figure(figsize=(8, 6))
        
        # vertices
        coordinates = {}
        for concept in self.concepts:
            x, y = np.array(sum([self.vectors[m] for m in self.intents[concept]], np.zeros(2)))
            coordinates[concept] = (x, -y)
            plt.scatter(x, -y, color="blue", zorder=3)
            plt.text(x, -y - 0.075 * self.N, concept, fontsize=12, ha='center', va='bottom', color='grey')

        # edges
        for (a, b) in cover_relations(self.lattice):
            x0, y0 = np.array(coordinates[a])
            x1, y1 = np.array(coordinates[b])
            plt.plot([x0, x1], [y0, y1], color="black", zorder=2)

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _sup_inf_distance(self):
        '''
        Supremum Infimum Distance for each combination of attributes.
        '''
        self.dsi_matrix = np.zeros((self.N, self.N))
        for i, j in combinations(range(self.N), 2):
            d = len(self.attribute_closures[self.attributes[i]] ^ self.attribute_closures[self.attributes[j]])
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
            for i, j in combinations(range(self.N), 2):
                # (|n_i - n_j| - d_SI(n_i, n_j))^2
                dist = np.linalg.norm(pts[i] - pts[j])
                e_si += (dist - self.dsi_matrix[i, j])**2
            return e_si

        res = minimize(e_si, np.random.rand(self.N * 2), method='BFGS')
        return res.x.reshape(-1, 2)

    def _initialize_vectors(self):
        '''
        Compute initial attribute vectors by minimizing the difference
        between the geometric distance and sup inf distance.
        '''
        # intial spring layout
        spring_pts = self._solve_spring_model()
        
        # longest path
        dists = squareform(pdist(spring_pts))
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        vec = spring_pts[j] - spring_pts[i]
        angle = -np.arctan2(vec[1], vec[0])
        
        # rotation
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        rotated_pts = spring_pts @ rot.T

        # initial attribute vectors        
        order = np.argsort(rotated_pts[:, 0])
        self.vectors = {}

        for i, idx in enumerate(order):
            x = -1 + 2 * (i / (self.N - 1))
            self.vectors[self.attributes[idx]] = np.array([x, x**2]) + np.random.normal(0, 0.01, size=2)

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
        indices = [self.attribute_map[m] for m in self.intents[concept]]
        # (0, 0) if concept has no attributes
        if not indices:
            return np.zeros(2)
        
        return np.sum(vectors[indices], axis=0)

    def _optimize_layout(self):
        '''
        Compute an optimized layout based on the initial spring layout.
        '''
        res = minimize(
            self._total_energy,
            np.array([self.vectors[m] for m in self.attributes]).flatten(),
            method='CG',
            options={'maxiter': 100, 'disp': True}
        )
        optimized_matrix = res.x.reshape(-1, 2)
        for i, m in enumerate(self.attributes):
            self.vectors[m] = optimized_matrix[i]

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
        Compute the Gravitational Energy, which constraints attribute vectors to "safe" angles
        to ensure an upward-directed, readable diagram.

        Parameters
        ----------
        flat_vectors : np.array
            A 1D flattened array of attribute vectors.

        Returns
        -------
        energy : float
            The gravitational energy.
        '''
        vectors = flat_vectors.reshape(-1, 2)
        
        # constants
        phi_0 = np.pi / (self.N + 1)
        E0 = -phi_0 - np.sin(phi_0) * np.cos(phi_0)
        E1 = (np.pi - phi_0) - np.sin(phi_0) * np.cos(phi_0)

        e_grav = 0.0
        for n_i in vectors:
            x, y = n_i
            phi = np.arctan2(y, x) # Returns (-pi, pi]

            # total Barrier for the lower half-plane (y <= 0)
            if phi <= 0:
                e_grav += 1e6 * (abs(phi) + 1.0)
                continue

            # Case 1: angle is too small
            if phi < phi_0:
                e_grav += phi + (np.cos(phi) / np.sin(phi)) * (np.sin(phi_0)**2) + E0

            # Case 2: angle is too large
            elif phi > (np.pi - phi_0):
                e_grav += -phi - (np.cos(phi) / np.sin(phi)) * (np.sin(phi_0)**2) + E1

        return e_grav