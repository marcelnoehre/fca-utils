import numpy as np
import networkx as nx

from typing import Iterable, Tuple
from fcapy.context import FormalContext

def from_covers(covers: Iterable[Tuple[int, int]], N: int) -> FormalContext:
    '''
    Create a formal context from a list of covering pairs.

    Parameters
    ----------
    covers : Iterable[Tuple[int, int]]
        An iterable of tuples representing the covering pairs (a, b).
    N : int
        The number of objects/attributes in the context.
    
    Returns
    -------
    context : FormalContext
        A formal context representing the given covering pairs.
    '''
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(N)])
    G.add_edges_from(covers)

    objects = [str(f'g{i}') for i in range(N)]
    atributes = [str(f'm{i}') for i in range(N)]
    incidence = [[(a == b or (a, b) in nx.transitive_closure(G).edges()) for a in range(N)] for b in range(N)]
    
    return FormalContext(object_names=objects, attribute_names=atributes, data=incidence)