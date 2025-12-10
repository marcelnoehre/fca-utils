import os
import sys
import warnings

# reference to src directory
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.linear_extension.realizer import *
from src.visualize.dim_draw import *

# compute 2-dimensional 'realizer' for the FM3
context = decode_cxt('../data/FM3.cxt')
lattice = ConceptLattice.from_context(context)

# compute 3-dimensional realizer for the FM3
dim, realizer = sat_realizer(lattice)
dim_draw = DimDraw(lattice, realizer)
dim_draw.plot_rotating_3d()