import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import deque 
from dataclasses import dataclass
from typing import Dict, Optional
from itertools import combinations
from scipy.optimize import minimize
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from scipy.spatial.distance import pdist, squareform

from fca.lattice import Lattice

@dataclass
class Args:
    si_graph: bool = False
    init_lat: bool = False
    opt_lat: bool = False
    annotations: bool = False
    forces: bool = False
    combined: bool = False

class FDP_Additive_Features():
    '''
    Force Directed Placement Algorithm to compute an attribute additive drawing of a
    Concept Lattice.

    Parameters
    ----------
    context : FormalContext
        The Formal Context
    name : str
        The name of the Formal Context
    args : Optional[Dict[str, bool]]
        A dictionary of arguments to customize the drawing.

    Reference
    ---------
    @inproceedings{ZschaligFDP,
        author = {Zschalig, Christian},
        booktitle = {CLA},
        publisher = {CEUR-WS.org},
        series = {CEUR Workshop Proceedings},
        title = {An FDP-Algorithm for Drawing Lattices},
        volume = 331,
        year = 2007
    }

    '''
    def __init__(self,
            context: FormalContext,
            name: str,
            args: Optional[Dict[str, bool]] = None
        ):
        self.name = name
        self.args: Args = Args(**(args or {}))
        self.context = context
        self.lattice = ConceptLattice.from_context(context)

        # attributes
        self.attributes = context.attribute_names
        self.attribute_map = context._attribute_names_i_map
        self.attribute_closures = {
            m: set(context.intention(context.extension([m])))
            for m in self.attributes
        }
        self.N_m = len(self.attributes)

        # concepts
        self.concepts = self.lattice.to_networkx().nodes
        self.intents = Lattice(context).all_intents()
        self.coatoms = [
            m for m in self.attributes
            if next((k for k, v in self.intents.items() if v == self.attribute_closures[m]), None) in self.lattice.children(0)
        ]
        self.cover_relations = Lattice(context).cover_relations()

        # weights
        self.w_rep = 30.0
        self.w_att = 1.0
        self.w_grav = 50.0

        # force directed placement
        self._sup_inf_distance()
        self._initialize_vectors()
        self._optimize_layout()
        

    def plot_d_si_graph(self, title):
        '''
        Plot the optimized and rotated d_SI Graph.

        Parameters
        ----------
        title: str
            The title for the figure.
        '''
        fig = plt.figure(figsize=(8, 6))
        fig.canvas.manager.set_window_title(title)

        # attributes as vertices
        for i, (x, y) in enumerate(self.d_si_points):
            plt.scatter(x, y, facecolor="white", edgecolor="black", linewidth=2.5, s=150, zorder=4)
            if self.args.annotations:
                plt.annotate(
                    self.attributes[i],
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    fontweight='bold'
                )

        # edges approximate d_SI(i, j)
        for i, j in combinations(range(self.N_m), 2):
            x_0, y_0 = np.array(self.d_si_points[i])
            x_1, y_1 = np.array(self.d_si_points[j])
            plt.plot([x_0, x_1], [y_0, y_1], color="black", linewidth=1, zorder=2)

        # f_max
        plt.plot([self.n_1[0], self.n_2[0]], [self.n_1[1], self.n_2[1]], color="black", linewidth=4, zorder=2)

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _sup_inf_distance(self):
        '''
        Supremum Infimum Distance for each combination of attributes.
        '''
        self.dsi_matrix = np.zeros((self.N_m, self.N_m))
        for i, j in combinations(range(self.N_m), 2):
            cl_i = self.attribute_closures[self.attributes[i]]
            cl_j = self.attribute_closures[self.attributes[j]]
            # incomparable
            if not (cl_i <= cl_j or cl_j <= cl_i):
                # m_i'' \cap m_j''
                sup = cl_i & cl_j
                # (m_i \cup m_j)''
                inf = set(self.context.intention(self.context.extension(cl_i | cl_j)))
                # |Inf| - |Sup| - 1
                d_si = len(inf) - len(sup) - 1
                self.dsi_matrix[i, j] = self.dsi_matrix[j, i] = d_si

    def _solve_spring_model(self):
        '''
        Compute initial spring model by minimizing the systems energy.
        '''
        def e_si(flat_positions):
            '''
            Compute the energy of the sup-inf system. 

            Parameters
            ----------
            flat_vectors : np.array
                A 1D flattened array of attribute vectors.
            '''
            attribute_positions = flat_positions.reshape(-1, 2)
            gradients = np.zeros_like(attribute_positions)
            
            e_si = 0.0
            for i, j in combinations(range(self.N_m), 2):
                n_i = attribute_positions[i]
                n_j = attribute_positions[j]

                # E_SI = (|n_i - n_j| - d_SI(n_i, n_j))^2
                e_si += (np.linalg.norm(n_i - n_j) - self.dsi_matrix[i, j])**2

            # F_SI(n_i)
            for i in range(self.N_m):
                for j in range(self.N_m):
                    if i == j:
                        continue
                    n_i = attribute_positions[i]
                    n_j = attribute_positions[j]

                    # F_SI = -2 * ((|n_i - n_j| - d_SI(n_i, n_j)) / |n_i - n_j|) * (n_i - n_j)
                    gradients[i] += -2 * ((np.linalg.norm(n_i - n_j) - self.dsi_matrix[i, j]) / np.linalg.norm(n_i - n_j)) * (n_i - n_j)

            # conjugate gradient (expects a negative gradient)
            return e_si, (gradients*-1).flatten()

        # intialize points at unit circle
        initial_pts = np.zeros((self.N_m, 2))
        for i in range(self.N_m):
            phi = 2 * np.pi * i / self.N_m
            initial_pts[i] = [np.cos(phi), np.sin(phi)]

        res = minimize(
            e_si,
            initial_pts.flatten(),
            method='CG',
            jac=True
        )
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

        # n_1 lies left of n_2
        if spring_pts[i, 0] < spring_pts[j, 0]:
            n_1, n_2 = spring_pts[i], spring_pts[j]
        else:
            n_1, n_2 = spring_pts[j], spring_pts[i]

        # rotated d_si points for plotting
        f_max = n_2 - n_1
        angle = -np.arctan2(f_max[1], f_max[0])
        
        # f_max horizontal 
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        self.d_si_points = spring_pts @ rot.T
        self.n_1 = n_1 @ rot.T
        self.n_2 = n_2 @ rot.T
        if self.args.si_graph:
            self.plot_d_si_graph(f'Sup-Inf_Graph: {self.name}')

        # (n_i - n_1) \cdot (n_2 - n_1)
        scalars = np.dot(spring_pts - n_1, n_2 - n_1)
        order = np.argsort(scalars)
        ordered_coatoms = sorted(self.coatoms, key=lambda c: scalars[self.attribute_map[c]])

        # place coatoms at parabola
        self.vectors = {}
        for i, m in enumerate(ordered_coatoms, start=1):
            x = round(1.8 * i + - 0.9 * (len(ordered_coatoms) + 1), 1)
            y = 0.09 * x**2 + 1.75
            self.vectors[m] = np.array([x, y])

        # remaining attributes that are no coatoms
        remaining = [m for m in self.attributes if m not in self.coatoms]
        queue = deque(remaining)
        upper = {
            m: self.attribute_closures[m].difference({m})
            for m in remaining
        }
        while queue:
            m = queue.popleft()
            # attributes > m
            if m in self.lattice.get_concept_new_intent(0):
                # \top = (0,0)
                self.vectors[m] = np.zeros(2)
            elif all(up in self.vectors for up in upper[m]):
                # handle all attributes with the same upper neighbors
                same_upper = [k for k, v in upper.items() if v == upper[m]]
                # sort based on scalar order derived from d_SI
                ordered_same_upper = [self.attributes[m] for m in order if m in [self.attribute_map[su] for su in same_upper]]
                # arithmetic mean of upper neighbors
                mean = sum([self.vectors[up] for up in upper[m]]) / len(upper[m])
                for i, m in enumerate(ordered_same_upper, start=1):
                    # offset based on scalar order
                    delta_i = np.array([1e-3 * i, 0])
                    self.vectors[m] = delta_i + mean
                # drop batch of processed attributes
                queue = deque([m for m in queue if m not in same_upper])
            else:
                queue.append(m)

        # derive coordinates for initial layout
        self.coordinates = {}
        for concept in self.concepts:
            x, y = np.array(sum([self.vectors[m] for m in self.intents[concept]], np.zeros(2)))
            # invert axes due to positive attribute vectors
            self.coordinates[concept] = (-x, -y)

        if self.args.init_lat:
            self.plot(f'Initial Layout: {self.name}')

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
            method='CG',
            jac=True,
            options={'maxiter': 1000}
        )

        # store optimized vectors
        optimized_matrix = res.x.reshape(-1, 2)
        for i, m in enumerate(self.attributes):
            self.vectors[m] = optimized_matrix[i]

        # derive coordinates for optimized layout
        self.coordinates = {}
        for concept in self.concepts:
            x, y = np.array(sum([self.vectors[m] for m in self.intents[concept]], np.zeros(2)))
            # invert axes due to positive attribute vectors
            self.coordinates[concept] = (-x, -y)

        if self.args.opt_lat:
            self.plot(f'Optimized Layout: {self.name}')

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
        energy = self.w_rep * e_rep + self.w_att * e_att + self.w_grav * e_grav
        gradients = self.w_rep * gradients_rep + self.w_att * gradients_att + self.w_grav * gradients_grav

        # conjugate gradient (expects a negative gradient)
        return energy, (gradients*-1).flatten()

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

        for v, w in enumerate(positions):
            for (i, j) in self.cover_relations:
                # edges without concept c
                if v == i or v == j:
                    continue
                
                # w = pos(c)
                # w_1 = pos(lower_edge_node)
                # w_2 = pos(upper_edge_node)

                # edge w_1, w_2 with w_1 below w_2
                if self.intents[i] <= self.intents[j]:
                    v_1, v_2 = j, i
                else:
                    v_1, v_2 = i, j
                w_1, w_2 = positions[v_1], positions[v_2]

                intent_v = self.intents[v]
                intent_v_1 = self.intents[v_1]
                intent_v_2 = self.intents[v_2]

                for m_i in self.attributes:
                    # attribute distribution
                    F_1 = (m_i not in intent_v) and (m_i not in intent_v_1) and (m_i not in intent_v_2)
                    F_2 = (m_i not in intent_v) and (m_i not in intent_v_1) and (m_i in intent_v_2)
                    F_3 = (m_i not in intent_v) and (m_i in intent_v_1) and (m_i not in intent_v_2)
                    F_4 = (m_i not in intent_v) and (m_i in intent_v_1) and (m_i in intent_v_2)
                    F_5 = (m_i in intent_v) and (m_i not in intent_v_1) and (m_i not in intent_v_2)
                    F_6 = (m_i in intent_v) and (m_i not in intent_v_1) and (m_i in intent_v_2)
                    F_7 = (m_i in intent_v) and (m_i in intent_v_1) and (m_i not in intent_v_2)
                    F_8 = (m_i in intent_v) and (m_i in intent_v_1) and (m_i in intent_v_2)

                    # not possible by order relation
                    if F_2 or F_6:
                        continue

                    # would not change conflict distance
                    if F_1 or F_8:
                        continue

                    # case 1:
                    # (w_1 - w) \cdot (w_2 - w_1) > 0
                    # concept w lies below w_1
                    if np.dot(w_1 - w, w_2 - w_1) > 0:
                        # |w_1 - w|
                        dist = np.linalg.norm(w_1 - w)

                        if F_3 or F_4:
                            # (1 / d(w, f)^2) * e(w_1 - w)
                            gradients_rep[self.attribute_map[m_i]] += (1 / dist**2) * ((w_1 - w) / dist)

                        if F_5:
                            # (1 / d(w, f)^2) * e(w - w_1)
                            gradients_rep[self.attribute_map[m_i]] += (1 / dist**2) * ((w - w_1) / dist)

                        # F_7 = 0

                    # case 2:
                    # (w_2 - w) \cdot (w_2 - w_1) < 0
                    # concept w lies above w_2
                    elif np.dot(w_2 - w, w_2 - w_1) < 0:
                        # |w_2 - w|
                        dist = np.linalg.norm(w_2 - w)

                        if F_4:
                            # (1 / d(w, f)^2) * e(w_2 - w)
                            gradients_rep[self.attribute_map[m_i]] += (1 / dist**2) * ((w_2 - w) / dist)

                        if F_5 or F_7:
                            # (1 / d(w, f)^2) * e(w - w_2)
                            gradients_rep[self.attribute_map[m_i]] += (1 / dist**2) * ((w - w_2) / dist)

                        # F_3 = 0

                    # case 3:
                    # (w_2 - w_1) \cdot (w - w_2) \leq 0, (w - w_1) \cdot (w - w_2) \geq 0
                    # concept w lies above w_1 and below w_2 
                    else:
                        # perpendicular distance
                        A = np.abs(np.cross(w_1 - w, w_2 - w))
                        f = w_2 - w_1
                        dist = A / np.linalg.norm(f)
                        
                        # (w_1 - w) \times (w_2 - w) \geq 0 -> w lies left of w_1w_2
                        # (w_1 - w) \times (w_2 - w) < 0 -> w lies right of w_1w_2
                        l = 1 if np.cross(w_1 - w, w_2 - w) >= 0 else -1

                        # n_+(f)
                        x_f, y_f = f
                        n_plus_f = np.array([-y_f, x_f])
                        h = A / np.linalg.norm(f)

                        if F_3:
                            # (1 / d(w, f)^2) * -sqrt(((w_2 - w)^2 - |h|^2) / |f|^2) * ((n_+(f) * l) / |f|)
                            gradients_rep[self.attribute_map[m_i]] += (1 / dist**2) * -np.sqrt((np.linalg.norm(w_2 - w)**2 - h**2) / np.linalg.norm(f)**2) * ((n_plus_f * l) / np.linalg.norm(f))

                        if F_4:
                            # (1 / d(w, f)^2) * -((n_+(f) * l) / |f|)
                            gradients_rep[self.attribute_map[m_i]] += (1 / dist**2) * -((n_plus_f * l) / np.linalg.norm(f))

                        if F_5:
                            # (1 / d(w, f)^2) * ((n_+(f) * l) / |f|)
                            gradients_rep[self.attribute_map[m_i]] += (1 / dist**2) * (n_plus_f * l) / np.linalg.norm(f)

                        if F_7:
                            # (1 / d(w, f)^2) * sqrt(((w_1 - w)^2 - |h|^2) / |f|^2) * ((n_+(f) * l) / |f|)
                            gradients_rep[self.attribute_map[m_i]] += (1 / dist**2) * np.sqrt((np.linalg.norm(w_1 - w)**2 - h**2) / np.linalg.norm(f)**2) * ((n_plus_f * l) / np.linalg.norm(f))

                # 1 / d(w, f)
                e_rep += 1.0 / dist

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


        for (i, j) in self.cover_relations:
            # w = pos(c)
            # w_1 = pos(lower_edge_node)
            # w_2 = pos(upper_edge_node)

            # edge w_1, w_2 with w_1 below w_2
            if self.intents[i] <= self.intents[j]:
                v_1, v_2 = j, i
            else:
                v_1, v_2 = i, j
            w_1, w_2 = positions[v_1], positions[v_2]

            intent_v_1 = self.intents[v_1]
            intent_v_2 = self.intents[v_2]
            
            for m_i in intent_v_1 - intent_v_2:
                # 2 * (w_2 - w_1)
                gradients_att[self.attribute_map[m_i]] += 2 * (w_2 - w_1)

            # |w_2 - w_1|^2
            e_att += np.sum((w_2 - w_1)**2)

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
        phi_0 = np.pi / (self.N_m + 1)

        # integration constants
        E_0 = -phi_0 - (np.sin(phi_0) * np.cos(phi_0))
        E_1 = E_0 + np.pi

        e_grav = 0.0
        gradients_grav = np.zeros_like(vectors)

        for m, (x, y) in enumerate(vectors):
            # angle of attribute vector in the range [-pi, pi]
            phi_m = np.arctan2(y, x)
            
            # 0 \leq phi_m \leq phi_0
            # angle too flat on the right side 
            if 0 <= phi_m <= phi_0:
                # E_grav(m) = phi_m + cot(phi_m) sin(phi_0)^2 + E_0
                e_grav += phi_m + (np.cos(phi_m) / (np.sin(phi_m))) * (np.sin(phi_0)**2) + E_0

            # pi - phi_0 \leq phi_m \leq pi 
            # angle too flat on the left side
            elif (np.pi - phi_0) < phi_m <= np.pi:
                # E_grav(m) = -phi_m - cot(phi_m) sin(phi_0)^2 + E_1
                e_grav += -phi_m - (np.cos(phi_m) / (np.sin(phi_m))) * (np.sin(phi_0)**2) + E_1

            if (0 <= phi_m <= phi_0) or ((np.pi - phi_0) < phi_m <= np.pi):
                x_m, y_m = vectors[m]

                # 1, if 0 < phi_m < phi_0
                if 0 < phi_m < phi_0:
                    direction = 1
                # -1, if phi_0 < phi_m < pi
                elif phi_0 < phi_m < np.pi:
                    direction = -1
                else:
                    direction = 0
                    
                # n_-(n_i) * ((sin(phi_m)^2 - sin(phi_0)^2) / y(n_i)^2) * direction
                gradients_grav[m] += np.array([y_m, -x_m]) * ((np.sin(phi_m)**2 - np.sin(phi_0)**2) / y_m**2) * direction

            # penalty for vectors pointing down
            if phi_m <= 0:
                penalty = 1e10
                # linear penalty based on how far below the axis it is
                e_grav += penalty * (abs(phi_m) + 1)
                # derivative of energy
                gradients_grav[m, 0] = 0.0
                gradients_grav[m, 1] = -penalty

        return e_grav, gradients_grav
    