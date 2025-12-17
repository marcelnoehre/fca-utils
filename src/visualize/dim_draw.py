import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from dataclasses import dataclass
from collections import defaultdict
from fcapy.lattice import ConceptLattice
from matplotlib.animation import FFMpegWriter
from typing import List, Dict, Optional, Tuple

from src.fca_utils.parser import *
from src.fca_utils.lattice import *

@dataclass
class Args:
    log: bool = False
    grid: bool = False
    concepts: bool = False
    indices: bool = False
    export: bool = False
    transform: bool = False

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
            realizer: Iterable[Iterable[int]]
        ):
        '''
        Initialize DimDraw with a given 'realizer'.

        Parameters
        ----------
        lattice : ConceptLattice
            The concept lattice.
        realizer : Iterable[Iterable[int]]
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

        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        if args.export:
            os.makedirs('output', exist_ok=True)
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
        if args.transform:
            # 3-dimensional roots
            w1 = np.exp(2j * np.pi / 3)
            w2 = np.exp(4j * np.pi / 3)

            # 3x3 transformation matrix using the real and imaginary parts
            A = np.array([
                [1, w1.real, w2.real],
                [0, w1.imag, w2.imag],
                [1, 1, 1]
            ])

            # coordinates as 3 x N array
            V = A @ np.array([self.coordinates[i] for i in self.nodes]).T  # 3 x N
            X, Y, Z = V

        else:
            # coordinates as 3 x N array
            V = np.array([self.coordinates[i] for i in self.nodes]).T  # 3 x N
            X, Y, Z = V

        ax.scatter(X, Y, Z, color='blue')
        ax.set_axis_off()

        # annotations
        for node in self.nodes:
            x, y, z = V[:, node]
            if args.concepts:
                ax.text(x, y, z + 0.075 * self.N, ','.join(self.lattice.get_concept_new_extent(node)), color='grey', fontsize=2 * self.N, zorder=10)
                ax.text(x, y, z - 0.075 * self.N, ','.join(self.lattice.get_concept_new_intent(node)), color='grey', fontsize=2 * self.N, zorder=10)
            elif args.indices:
                ax.text(x, y, z - 0.075 * self.N, node, color='grey', fontsize=2 * self.N, zorder=10)

        # connect concepts based on relations
        for i, j in cover_relations(self.lattice):
            xi, yi, zi = V[:, i]
            xj, yj, zj = V[:, j]
            ax.plot([xi, xj], [yi, yj], [zi, zj], 'k-', color='gray', alpha=0.5)
            
        if args.export:
            os.makedirs('output', exist_ok=True)
            # capture frames for video
            writer = FFMpegWriter(fps=30)
            with writer.saving(fig, 'output/rotating_dim_draw.mp4', dpi=200):
                for azim in range(0, 360 * 4 + 1):
                    ax.view_init(elev=0, azim=azim)
                    writer.grab_frame()

            plt.close(fig)
        else:
            # rotate interactive view
            for azim in range(0, 360*4 + 1):
                if not plt.fignum_exists(fig.number):
                    break
                ax.view_init(elev=0, azim=azim)
                plt.draw()
                plt.pause(0.01)

            plt.show()
