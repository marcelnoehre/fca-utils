from collections import defaultdict
from fcapy.lattice import ConceptLattice

from src.fca_utils.parser import *
from src.fca_utils.lattice import *

class DimDraw():
    '''
    Disclaimer
    ----------
    This Python script is based on the DimDraw originally developed by Prof. Dr. Dominik Dürrschnabel.
    This code is an independent visualization.
    The original version is integrated into the tool conexp-clj [see https://github.com/tomhanika/conexp-clj].

    Reference
    ---------
    @misc{dürrschnabel2019dimdrawnoveltool,
        title={DimDraw -- A novel tool for drawing concept lattices},
        author={Dominik Dürrschnabel and Tom Hanika and Gerd Stumme},
        year={2019},
        eprint={1903.00686},
        archivePrefix={arXiv},
        primaryClass={cs.CG},
        url={https://arxiv.org/abs/1903.00686}
    }
    '''

    def __init__(self,
            lattice: ConceptLattice,
            realizer: Tuple[Iterable[int], Iterable[int]]
        ):
        '''
        Initialize DimDraw with a given 'realizer'.

        Parameters
        ----------
        lattice : ConceptLattice
            The concept lattice.
        realizer : Tuple[Iterable[int], Iterable[int]]
            A 'realizer' defining the DimDraw axes
        '''
        self.lattice = lattice
        self.nodes = self.lattice.to_networkx().nodes
        self.N = len(self.nodes)
        self.dimension = len(realizer)
        self.realizer = tuple(
            realizer[i] if realizer[i][0] == 0 else list(reversed(realizer[i])) 
            for i in range(self.dimension)
        )
        self.objects = extent_of_concept(self.lattice, self.realizer[0][0])
        self.features = intent_of_concept(self.lattice, self.realizer[0][-1])
        self._compute_coordinates()
        if self.dimension == 2:
            self._setup_grid_2d()
        else:
            raise ValueError(f'Dimension {self.dimension} is not implemented so far!')

    def _setup_grid_2d(self):
        '''
        Setup nodes and connections that form the 2-dimensional DimDraw grid 
        '''
        self.grid = defaultdict(list)
        self.grid[self.realizer[0][0]].append((0, 0))
        self.connections = []

        ext1, ext2 = (reversed(r[-1:] + r[1:-1]) for r in self.realizer[:2])
        prev_x, prev_y = (0, 0)

        for i, (node_x, node_y) in enumerate(zip(ext1, ext2)):
            # horizontal
            self.grid[node_x].append((i + 1, 0))
            self.connections.append(((prev_x, 0), (i + 1, 0)))
            prev_x = i + 1
            # vertical
            self.grid[node_y].append((0, i + 1))
            self.connections.append(((0, prev_y), (0, i + 1)))
            prev_y = i + 1

        ext1, ext2 = (reversed(r[:-1]) for r in self.realizer[:2])
        self.connections.append(((self.N - 1, 0), (self.N - 1, 1)))
        self.connections.append(((0, self.N - 1), (1, self.N - 1)))
        for i, (node_x, node_y) in enumerate(zip(ext1, ext2)):
            # horizontal
            self.grid[node_x].append((i + 1, self.N - 1))
            self.connections.append(((prev_x, self.N - 1), (i + 1, self.N - 1)))
            prev_x = i + 1
            # vertical
            self.grid[node_y].append((self.N - 1, i + 1))
            self.connections.append(((self.N - 1, prev_y), (self.N - 1, i + 1)))
            prev_y = i + 1

    def _compute_coordinates(self):
        '''
        Compute the coordinates for concepts based on their rank in the linear extensions
        '''
        self.coordinates = {
            node: tuple(list(reversed(le)).index(node) for le in self.realizer)
            for node in self.nodes
        }