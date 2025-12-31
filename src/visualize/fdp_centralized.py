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
        # for _ in range(repetitions):
        #     self._initialize_vectors()
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
