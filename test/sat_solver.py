import os
import sys
import warnings

# reference to src directory
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.linear_extension.realizer import *
from src.visualize.dim_draw import *
from src.additivity.additivity import AdditivityCheck

context = decode_cxt('../data/FM3.cxt')
lattice = ConceptLattice.from_context(context)

# compute 3-dimensional realizer for the FM3
dim, realizer = sat_realizer(lattice)

dim_draw = DimDraw(lattice, realizer)
dim_draw.plot_rotating_3d()

check = AdditivityCheck(lattice, realizer, dim_draw.coordinates)
print(f'\x1b[33mBottom-Up Addiive:\x1b[0m {check.check_bottom_up_additivity()}')
print(f'\x1b[33mTop-Down Additive:\x1b[0m {check.check_top_down_additivity()}')
print(f'\x1b[33mCombined Additive:\x1b[0m {check.check_combined_additivity()}')

print('\n\x1b[33mLinear Equations (Combined Additivity):\x1b[0m')
for eq in check.combined_equations:
    print(eq)