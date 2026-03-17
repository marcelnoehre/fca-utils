import re
import subprocess

from typing import Tuple, Iterable

from fcapy.lattice import ConceptLattice

from data.parser import Parser
from fca.context import Context
from fca.lattice import Lattice

class DimDraw():
    '''
    This script uses the original version of DimDraw integrated into the tool conexp-clj [see https://github.com/tomhanika/conexp-clj].

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

    def __init__(self, cxt: str, brunt: str):
        '''
        Initialize DimDraw with a given 'realizer'.

        Parameters
        ----------
        cxt : str
            Path to a context file.
        brunt : str
            Path to the Brunt library.
        '''
        self.cxt_path = cxt
        self.brunt_path = brunt
        parser = Parser()
        self.context = Context(parser.decode_cxt(self.cxt_path))
        self.lattice = Lattice(self.context.context)
        self.concepts = self.lattice.nodes
        self.realizer = self.two_dimensional_extension()
        self.compute_coordinates()

    def two_dimensional_extension(self) -> Tuple[Iterable[int], Iterable[int]]:
        '''
        Compute the 2-dimensional 'realizer' using the conexp-clj API

        Returns
        -------
        realizer : Tuple[Iterable[int], Iterable[int]]
            The two-dimensional extension for DimDraw drawings
        '''
        N = len(self.concepts)
        le_x, le_y = [None] * N, [None] * N

        # extent and intent for each node
        concepts = {}
        for node in self.concepts:
            concepts[node] = {
                g for _,g in self.lattice.extent_of_concept(node)
            }.union({
                m for _,m in self.lattice.intent_of_concept(node)
            })

        try:
            # execute conexp-clj
            res = subprocess.check_output([
                "java", "-jar", self.brunt_path, 
                '-f', 'dim-draw-coordinates',
                self.cxt_path], text=True)
            
            # derive realizer position from coordinates
            for line in res.splitlines():
                concept, coords = line.split(' -> ')
                x, y = coords.strip("()").split(", ")
                node = next(
                    (k for k, v in concepts.items() 
                    if v == set(re.findall(r"\b[g|m][A-Za-z0-9]+\b", concept))),None
                )
                le_x[int(x)] = node
                le_y[int(y)] = node

            return (le_x, le_y)

        except subprocess.CalledProcessError as e:
            print(f"Error running JAR: {e}")

    def compute_coordinates(self):
        '''
        Compute the coordinates for concepts based on their rank in the linear extensions
        '''
        self.positions = {
            node: tuple(list(reversed(le)).index(node) for le in self.realizer)
            for node in self.concepts
        }
