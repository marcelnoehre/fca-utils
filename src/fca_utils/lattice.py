import copy
import networkx as nx

from collections import deque
from itertools import combinations
from fcapy.lattice import ConceptLattice
from typing import Iterable, Tuple, List, Set

def incomparability_graph(lattice: ConceptLattice) -> nx.Graph:
    '''
    Get the incomparability graph of a concept lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.
    
    Returns
    -------
    incomparability_graph : nx.Graph
        The incomparability graph of the lattice.
    '''
    return nx.complement(nx.transitive_closure(lattice.to_networkx()).to_undirected())

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

def minimal_realizers(lattice: ConceptLattice, linear_extensions: Iterable[Tuple[int]], all: bool = True):
    '''
    Find all minimal realizers from a set of linear extensions.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.
    linear_extensions : Iterable[Tuple[int]]
        An iterable of linear extensions of the lattice.
    all : bool, optional
        Whether to find all minimal realizers or stop after the first one is found. Default is True.

    Returns
    -------
    dimension : int
        The order dimension of the lattice (size of the minimal realizer).
    minimal_realizers : Set[Iterable[Tuple[int]]]
        A set of all minimal realizers found or the first one if 'all' is False.
    '''
    # find all pairs of incomparable concepts
    incompareables = list(incomparability_graph(lattice).edges())
    k = 1 if len(incompareables) == 0 else 2
    minimal_realizers = set()

    def _check_realizer(subset: Iterable[Tuple[int]]):
        '''
        Check if a subset of linear extensions forms a realizer.

        Parameters
        ----------
        subset : Iterable[Tuple[int]]
            A subset of linear extensions.
        
        Returns
        -------
        realizer : Set[Tuple[int]] | None
            The realizer if the subset forms a realizer, None otherwise.
        '''
        for (x, y) in incompareables:
            # check if all incomparables are reversed in at least one linear extension
            fwd, bwd = False, False
            for linear_extension in subset:
                if linear_extension.index(x) < linear_extension.index(y):
                    fwd = True
                if linear_extension.index(y) < linear_extension.index(x):
                    bwd = True
            
            if not fwd or not bwd:
                return None
            
        return set(subset)
    
    while not minimal_realizers and k <= len(linear_extensions):
        #iterate over all subsets of linear extensions of size k
        for subset in combinations(linear_extensions, k):
            realizer = _check_realizer(subset)
            
            # store valid realizer
            if realizer:
                minimal_realizers.add(copy.deepcopy(tuple(realizer)))

                # search for all minimal realizers
                if not all:
                    break
        
        # no valid realizer found, increase number of linear extensions in subset
        if not minimal_realizers:
            k += 1

    return k, minimal_realizers

def all_children(lattice: ConceptLattice, index: int) -> Set[int]:
    '''
    Get all children of a concept in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.
    index : int
        The index of the concept.

    Returns
    -------
    children : Set[int]
        A set of indices representing all children of the concept.
    '''
    visited = set()
    queue = deque([index])
    children = set()

    while queue:
        node = queue.popleft()
        
        for child in lattice.children(node):
            if child not in visited:
                visited.add(child)
                children.add(child)
                queue.append(child)

    return children

def all_parents(lattice: ConceptLattice, index: int) -> Set[int]:
    '''
    Get all parents of a concept in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.
    index : int
        The index of the concept.

    Returns
    -------
    parents : Set[int]
        A set of indices representing all parents of the concept.
    '''
    visited = set()
    queue = deque([index])
    parents = set()

    while queue:
        node = queue.popleft()
        for parent in lattice.parents(node):
            if parent not in visited:
                visited.add(parent)
                parents.add(parent)
                queue.append(parent)

    return parents

def intent_of_concept(lattice: ConceptLattice, index: int) -> Set[int]:
    '''
    Get the intent of a concept in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.
    index : int
        The index of the concept.

    Returns
    -------
    intent : Set[int]
        A set of indices representing the intent of the concept.
    '''
    parents = all_parents(lattice, index)
    intent = set()
    for feature in lattice.get_concept_new_intent(index):
        intent.add((index, feature))

    for parent in parents:
        for feature in lattice.get_concept_new_intent(parent):
            intent.add((parent, feature))
    
    return intent

def extent_of_concept(lattice: ConceptLattice, index: int) -> Set[int]:
    '''
    Get the extent of a concept in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.
    index : int
        The index of the concept.

    Returns
    -------
    extent : Set[int]
        A set of indices representing the extent of the concept.
    '''
    children = all_children(lattice, index)
    extent = set()
    for obj in lattice.get_concept_new_extent(index):
        extent.add((index, obj))

    for child in children:
        for obj in lattice.get_concept_new_extent(child):
            extent.add((child, obj))
            
    return extent