import networkx as nx

from fcapy.lattice import ConceptLattice
from typing import Iterable, Tuple

def linear_extensions_topological(lattice: ConceptLattice) -> Iterable[Tuple[int]]:
    '''
    Topological sorting of the concept lattice to get all linear extensions.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.

    Returns
    -------
    linear_extensions : Iterable[Tuple[int]]
        An iterable of all linear extensions of the lattice.

    Notes
    -----
    Only suitable for small lattices due to combinatorial explosion.
    '''
    return {tuple(le) for le in nx.all_topological_sorts(lattice.to_networkx())}