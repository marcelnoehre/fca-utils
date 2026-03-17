from data.parser import Parser
from draw.fdp import FDP_Additive_Features
from visualize.plot import *

parser = Parser()
context = parser.decode_cxt('./data/standard/B3.cxt')
dim_draw = FDP_Additive_Features(context, 'B3', {})

plot_lattice(context, dim_draw.concepts, dim_draw.coordinates)