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
        self.object_map = {name: i + len(self.attributes) for i, name in enumerate(self.attributes)}
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
        #     self._optimize_layout()

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
            x = -1 + 2 * (i / (self.N_m + self.N_g - 1))
            if idx < self.N_m:
                self.vectors[self.attributes[idx]] = np.array([x, x**2]) + np.random.normal(0, 0.01, size=2)
            else:
                self.vectors[self.objects[idx - self.N_m]] = np.array([x, x**2]) + np.random.normal(0, 0.01, size=2)