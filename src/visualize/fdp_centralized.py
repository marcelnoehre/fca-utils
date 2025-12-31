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
            self._optimize_layout()