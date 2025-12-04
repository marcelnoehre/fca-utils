from typing import Set
from collections import deque
from fcapy.lattice import ConceptLattice

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
    Set[int]
        A set of indices representing all children of the concept.
    '''
    visited = set()
    queue = deque([index])
    result = set()

    while queue:
        node = queue.popleft()
        
        for child in lattice.children(node):
            if child not in visited:
                visited.add(child)
                result.add(child)
                queue.append(child)

    return result

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
    Set[int]
        A set of indices representing all parents of the concept.
    '''
    visited = set()
    queue = deque([index])
    result = set()

    while queue:
        node = queue.popleft()
        for parent in lattice.parents(node):
            if parent not in visited:
                visited.add(parent)
                result.add(parent)
                queue.append(parent)

    return result