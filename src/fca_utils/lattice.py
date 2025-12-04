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