import os
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from collections import defaultdict
from fcapy.lattice import ConceptLattice
from typing import List, Dict, Optional, Tuple

from src.fca_utils.parser import *
from src.fca_utils.lattice import *

@dataclass
class Args:
    log: bool = False
    grid: bool = False
    concepts: bool = False
    indices: bool = False

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
        elif self.dimension == 3:
            pass
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

    def _plot_lattice(self,
            filename: str,
            coordinates: Dict[int, Tuple[int, int]],
            relations: List[Tuple[Tuple[int, int], Tuple[int, int]]],
            highlight_nodes: List[int] = [],
            args: Optional[Dict[str, bool]] = None
        ):
        plt.figure(figsize=(8, 6))
        args: Args = Args(**(args or {}))

        if args.log:
            for node in self.nodes:
                print(f'\x1b[35mNode {node}:\x1b[0m')
                print(f'New extent: \x1b[33m{",".join(self.lattice.get_concept_new_extent(node))}\x1b[0m')
                print(f'New intent: \x1b[33m{",".join(self.lattice.get_concept_new_intent(node))}\x1b[0m')

        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
        
        # Grid
        if args.grid:
            # grid nodes
            for node, positions in self.grid.items():
                for pos in positions:
                    x, y = R @ np.array(pos)
                    plt.scatter(x, y, color="lightgrey", zorder=1)
                    plt.text(x, y, node, fontsize=12, color='grey')

            # connect grid nodes
            for connection in self.connections:
                x0, y0 = R @ np.array(connection[0])
                x1, y1 = R @ np.array(connection[1])
                plt.plot([x0, x1], [y0, y1], color="lightgrey", zorder=1)

        # concepts
        for node, coordinate in coordinates.items():
            x, y = R @ np.array(coordinate)
            plt.scatter(x, y, color="orange" if node in highlight_nodes else "blue", zorder=3)

            # annotations
            if args.concepts:
                plt.text(x, y + 0.075 * self.N, ','.join(self.lattice.get_concept_new_extent(node)), fontsize=12, ha='center', va='top', color='grey')
                plt.text(x, y - 0.075 * self.N, ','.join(self.lattice.get_concept_new_intent(node)), fontsize=12, ha='center', va='bottom', color='grey')
            elif args.indices:
                plt.text(x, y - 0.075 * self.N, node, fontsize=12, ha='center', va='bottom', color='grey')

        # connect concepts based on relations
        for relation in relations:
            x0, y0 = R @ np.array(relation[0])
            x1, y1 = R @ np.array(relation[1])
            plt.plot([x0, x1], [y0, y1], color="black", zorder=2)

        os.makedirs('output', exist_ok=True)
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f'output/{filename}')
        plt.show()

    def plot(self,
            highlight_nodes: List[int] = [],
            args: Optional[Dict[str, bool]] = None
        ):
        '''
        Draw the concept lattice using DimDraw.

        Parameters
        ----------
        highlight_nodes : List[int]
            A list of nodes to highlight in the drawing.

        args : Optional[Dict[str, bool]]
            A dictionary of arguments to customize the drawing.
        '''
        if self.dimension != 2:
            raise ValueError('2D plotting is only available for 2-dimensional realizers!')
        
        self._plot_lattice(
            'dim_draw.png',
            self.coordinates,
            [(self.coordinates[a], self.coordinates[b]) for a, b in cover_relations(self.lattice)],
            highlight_nodes,
            args
        )

    def plot_rotating_3d(self,
            args: Optional[Dict[str, bool]] = None
        ):
        '''
        Plot a rotating 3D visualization of the concept lattice.

        Parameters
        ----------
        args : Optional[Dict[str, bool]]
            A dictionary of arguments to customize the drawing.
        '''
        if self.dimension != 3:
            raise ValueError('3D plotting is only available for 3-dimensional realizers!')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        args: Args = Args(**(args or {}))

        # concepts
        X = np.array([coord[0] for coord in self.coordinates.values()])
        Y = np.array([coord[1] for coord in self.coordinates.values()])
        Z = np.array([coord[2] for coord in self.coordinates.values()])
        ax.scatter(X, Y, Z, color='blue')

        # annotations
        for node, (x, y, z) in self.coordinates.items():
            if args.concepts:
                ax.text(x, y, z + 0.075 * self.N, ','.join(self.lattice.get_concept_new_extent(node)), color='grey', fontsize=2 * self.N, zorder=10)
                ax.text(x, y, z - 0.075 * self.N, ','.join(self.lattice.get_concept_new_intent(node)), color='grey', fontsize=2 * self.N, zorder=10)
            elif args.indices:
                ax.text(x, y, z - 0.075 * self.N, node, color='grey', fontsize=2 * self.N, zorder=10)

        # connect concepts based on relations
        relations = [(self.coordinates[a], self.coordinates[b]) for a, b in cover_relations(self.lattice)]
        for (x0, y0, z0), (x1, y1, z1) in relations:
            ax.plot([x0, x1], [y0, y1], [z0, z1], color='gray', alpha=0.5)

        # rotate view
        for azim in range(0, 360*4 + 1):
            if not plt.fignum_exists(fig.number):
                break
            ax.view_init(elev=30, azim=azim)
            plt.draw()
            plt.pause(0.01)

        plt.show()