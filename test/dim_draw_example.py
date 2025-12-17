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
context = decode_cxt('../data/standard_lattices/FM3.cxt')
lattice = ConceptLattice.from_context(context)
realizer = dim_draw_realizer('../data/standard_lattices/FM3.cxt', '../libraries/brunt-fork.jar')

# plot DimDraw layout
dim_draw = DimDraw(lattice, realizer)
dim_draw.plot(args={'concepts': True})
