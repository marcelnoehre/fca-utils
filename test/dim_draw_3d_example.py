import os
import sys
import warnings

# reference to src directory
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.linear_extension.linear_extension import *
from src.linear_extension.realizer import *
from src.visualize.dim_draw import *

# compute 3-dimensional realizer for the cube lattice
context = decode_cxt('../data/cube.cxt')
lattice = ConceptLattice.from_context(context)
linear_extensions = linear_extensions_topological(lattice)
dim, realizers = minimal_realizers(lattice, linear_extensions, False)

# plot rotating 3D DimDraw layout
dim_draw = DimDraw(lattice, list(realizers)[0])
dim_draw.plot_rotating_3d({'concepts': True})
