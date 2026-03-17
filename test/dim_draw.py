from data.parser import Parser
from draw.dim_draw import DimDraw
from visualize.plot import *

parser = Parser()
context = parser.decode_cxt('./data/standard/B3.cxt')
dim_draw = DimDraw('./data/standard/B3.cxt', './libs/brunt-fork.jar')

plot_lattice(context, dim_draw.concepts, dim_draw.positions, './test.tex')