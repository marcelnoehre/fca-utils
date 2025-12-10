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
        self.realizer = tuple(
            realizer[i] if realizer[i][0] == 0 else list(reversed(realizer[i])) 
            for i in range(len(realizer))
        )
        
        self.objects = extent_of_concept(self.lattice, self.realizer[0][0])
        self.features = intent_of_concept(self.lattice, self.realizer[0][-1])
