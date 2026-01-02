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
        self.attributes = self.context.attribute_names
        self.attribute_map = {name: i for i, name in enumerate(self.attributes)}
        self.attribute_closures = {
            m: set(context.intention(context.extension([m])))
            for m in self.attributes
        }
        self.score = np.inf
        self.N = len(self.attributes)
        self.concepts = self.lattice.to_networkx().nodes
        self.intents = {
            concept: [val for _, val in intent_of_concept(self.lattice, concept)]
            for concept in self.concepts
        }
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
            plt.text(x, y - 0.075 * self.N, concept, fontsize=12, ha='center', va='bottom', color='grey')

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
            fun=self._total_energy_and_gradient,
            x0=np.array([self.vectors[m] for m in self.attributes]).flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'disp': True}
        )
        optimized_matrix = res.x.reshape(-1, 2)
        for i, m in enumerate(self.attributes):
            self.vectors[m] = optimized_matrix[i]
        
        if res.fun < self.score:
            self.coordinates = {}
            for concept in self.concepts:
                x, y = np.array(sum([self.vectors[m] for m in self.intents[concept]], np.zeros(2)))
                self.coordinates[concept] = (x, y)

    def _total_energy_and_gradient(self, flat_vectors):
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
        e_rep, gradients_rep = self._repulsive_force(flat_vectors)
        e_att, gradients_att = self._attractive_force(flat_vectors)
        e_grav, gradients_grav = self._gravitational_force(flat_vectors)

        # weights
        rep = 1.0
        att = 1.0
        grav = 1.0

        energy = rep * e_rep + att * e_att + grav * e_grav
        gradients = rep * gradients_rep + att * gradients_att + grav * gradients_grav
        return energy, gradients.flatten()

    def _repulsive_force(self, flat_vectors):
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

        e_rep = 0.0
        gradients_rep = np.zeros_like(vectors)

        for c, w in enumerate(positions):
            for (a, b) in cover_relations(self.lattice):
                # edges without concept c
                if c == a or c == b:
                    continue
                
                # w = pos(c)
                # w_1 = pos(lower_edge_node)
                # w_2 = pos(upper_edge_node)

                # edge w_1, w_2 with w_1 below w_2
                if set(self.intents[a]).issubset(set(self.intents[b])):
                    w_1, w_2 = positions[b], positions[a]
                else:
                    w_1, w_2 = positions[a], positions[b]

                # case 1:
                # (w_1 - w) \cdot (w_2 - w_1) > 0
                # concept w lies below w_1
                if np.dot(w_1 - w, w_2 - w_1) > 0:
                    # distance of concept w to lower point w_1
                    dist = np.linalg.norm(w - w_1)
                    # force from w_1 to w
                    force_direction = w - w_1

                # case 2:
                # (w_2 - w) \cdot (w_2 - w_1) < 0
                # concept w lies above w_2
                elif np.dot(w_2 - w, w_2 - w_1) < 0:
                    # distance of concept w to upper point w_2
                    dist = np.linalg.norm(w - w_2)
                    # force from w_2 to w 
                    force_direction = w - w_2

                # case 3:
                # (w_2 - w_1) \cdot (w - w_2) \leq 0, (w - w_1) \cdot (w - w_2) \geq 0
                # concept w lies above w_1 and below w_2 
                else:
                    # perpendicular distance
                    # |(w_1 - w) \times (w_2 - w)| / |w_2 - w_1|
                    dist = np.abs(np.cross(w_1 - w, w_2 - w)) / np.linalg.norm(w_2 - w_1)
                    # force from edge to concept w
                    v = w_2 - w_1 # edge
                    t = np.dot(w - w_1, v) / np.dot(v, v) # scalar projection parameter
                    p = w_1 + t * v # projection point (Lotfußpunkt)
                    force_direction = w - p

                epsilon = 1e-6
                f_mag = 1.0 / (dist + epsilon)**2
                f_vec = (force_direction / (dist + epsilon)) * f_mag

                intent = [self.attribute_map[m] for m in self.intents[c]]
                for m in intent:
                    gradients_rep[m] -= f_vec / len(intent)

                e_rep += 1.0 / (dist + epsilon)

        return e_rep, gradients_rep
    
    def _attractive_force(self, flat_vectors):
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
        gradients_att = np.zeros_like(vectors)

        for (a, b) in cover_relations(self.lattice):
            diff = positions[b] - positions[a]
            
            intent_a = [self.attribute_map[m] for m in self.intents[a]]
            for m in intent_a:
                gradients_att[m] += 2 * diff

            intent_b = [self.attribute_map[m] for m in self.intents[b]]
            for m in intent_b:
                gradients_att[m] -= 2 * diff

            # |pos(b) - pos(a)|^2
            e_att += np.sum(diff**2)

        return e_att, gradients_att
    
    def _gravitational_force(self, flat_vectors):
        """
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
        """
        vectors = flat_vectors.reshape(-1, 2)

        # angle phi_0
        phi_0 = np.pi / (len(self.attributes) + 1)

        # integration constants
        E_0 = -phi_0 - (np.sin(phi_0) * np.cos(phi_0))
        E_1 = E_0 + np.pi

        e_grav = 0.0
        gradients_grav = np.zeros_like(vectors)
        epsilon = 1e-12

        for m, (x, y) in enumerate(vectors):
            # angle of attribute vector in the range [-pi, pi]
            phi_m = np.arctan2(y, x)

            # square of the radius for the chain rule derivations
            r_sq = x**2 + y**2 + epsilon

            # case 1:
            # 0 \leq phi_m \leq phi_0
            # angle too flat on the right side 
            if 0 <= phi_m <= phi_0:
                # E_grav(m) = phi_m + cot(phi_m) sin(phi_0)^2 + E_0
                e_grav += phi_m + (np.cos(phi_m) / (np.sin(phi_m) + epsilon)) * (np.sin(phi_0)**2) + E_0

                # derivation of energy by angle d_E / d_phi_m
                # (sin^2(phi_m) - sin^2(phi_0)) / sin^2(phi_m)
                d_E_phi = (np.sin(phi_m)**2 - np.sin(phi_0)**2) / (np.sin(phi_m)**2 + epsilon)

                # chain rule: dE/dx = dE/dphi * dphi/dx and dE/dy = dE/dphi * dphi/dy
                # d_phi/dx = -y / r^2 
                # d_phi/dy = x / r^2
                gradients_grav[m, 0] = d_E_phi * (-y / r_sq)
                gradients_grav[m, 1] = d_E_phi * (x / r_sq)

            # case 2:
            # pi - phi_0 \leq phi_m \leq pi 
            # angle too flat on the left side
            elif (np.pi - phi_0) < phi_m <= np.pi:
                # E_grav(m) = -phi_m - cot(phi_m) sin(phi_0)^2 + E_1
                e_grav += -phi_m - (np.cos(phi_m) / (np.sin(phi_m) + epsilon)) * (np.sin(phi_0)**2) + E_1
                
                # derivation: d_E / d_phi_m
                # factor -1.0 accounts for the mirrored slope
                d_E_dphi = -1.0 * (np.sin(phi_m)**2 - np.sin(phi_0)**2) / (np.sin(phi_m)**2 + epsilon)
                
                # chain rule: dE/dx = dE/dphi * dphi/dx and dE/dy = dE/dphi * dphi/dy
                # d_phi/dx = -y / r^2 
                # d_phi/dy = x / r^2
                gradients_grav[m, 0] = d_E_dphi * (-y / r_sq)
                gradients_grav[m, 1] = d_E_dphi * (x / r_sq)

            # case 3:
            # penalty for vectors into the wrong direction
            elif phi_m <= 0:
                penalty = 1e3
                # linear penalty based on how far below the axis it is
                e_grav += penalty * (abs(phi_m) + 1)
                # derivative of energy
                gradients_grav[m, 0] = 0.0
                gradients_grav[m, 1] = -penalty

        return e_grav, gradients_grav
    