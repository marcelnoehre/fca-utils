import os
import sys
import warnings

# reference to src directory
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.additivity.additive_realizer import *
from src.visualize.dim_draw import *

context = decode_cxt('../data/N5.cxt')
lattice = ConceptLattice.from_context(context)

# compute 3-dimensional realizer for the FM3
dim, realizer = AdditiveRealizer(lattice).realizer()