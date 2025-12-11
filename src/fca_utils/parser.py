import math

from sage.all import Poset
from typing import Iterable, Tuple
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

from src.fca_utils.context import from_covers

def decode_lce(lce: str) -> Iterable[Tuple[int, int]]:
    '''
    Decode a levellised covering encoding (LCE) string into a Formal Context.
    
    Each character corresponds to an entry in the upper-triangular part of the incidence matrix.

    '1' indicates the presence of a covering pair (a, b), while any other character indicates its
    absence.

    Parameters
    ----------
    lce : str
        A string representing the levellised covering encoding or a path to the .lce file

    Returns
    -------
    formal_context : FormalContext
        The formal context.
    '''
    if lce.endswith('.lce'):
        with open(lce, 'r') as f:
            lce = f.read()

    from_covers([pair
        for i, pair in enumerate(([(a, b) 
            for b in range(1, int(0.5 + math.sqrt(0.25 + 2 * len(lce)))) 
            for a in range(b)])) 
        if lce[i] == '1'
    ])

def decode_cxt(cxt: str) -> FormalContext:
    '''
    Decode a Burmeister (B) string into a Formal Context.

    The string starts with a B, followed by the dimension of the context and the incidence matrix.

    'x' or 'X' indicates that a object (row) has a feature (column), while a any other character
    indicates that a object does not have a feature. 

    Parameters
    ----------
    cxt : str
        A string representing the burmeister format or a path to the .cxt file

    Returns
    -------
    formal_context : FormalContext
        The formal context.
    '''
    if cxt.endswith('.cxt'):
        with open(cxt, 'r') as f:
            cxt = f.read()

    _, ns, cxt = cxt.split('\n\n')
    n_objs, n_attrs = [int(x) for x in ns.split('\n')]

    cxt = cxt.strip().split('\n')
    obj_names, cxt = cxt[:n_objs], cxt[n_objs:]
    attr_names, cxt = cxt[:n_attrs], cxt[n_attrs:]
    cxt = [[(c == 'X' or c == 'x') for c in line] for line in cxt]

    return FormalContext(data=cxt, object_names=obj_names, attribute_names=attr_names)

def sage_poset_from_lattice(lattice: ConceptLattice) -> Poset:
    '''
    Convert a FcaPy ConceptLattice into a SageMath Poset.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice to convert.

    Returns
    -------
    poset : Poset
        The corresponding SageMath Poset.
    '''
    nodes = list(lattice.to_networkx().nodes)
    return Poset({
        node: list(lattice.parents(node)) 
        for node in nodes
    })