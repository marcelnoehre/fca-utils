import os
import sys
import warnings

# reference to src directory
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('src'), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.fca_utils.parser import decode_cxt
from src.visualize.fdp_m import FDP_Additive_Features

context = decode_cxt('../data/standard_lattices/B3.cxt')

fdp = FDP_Additive_Features(context)
fdp.plot()