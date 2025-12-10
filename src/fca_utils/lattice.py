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

def get_join_irreducibles(lattice: ConceptLattice) -> Iterable[int]:
    '''
    Get all join-irreducible concepts in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.

    Returns
    -------
    join_irreducibles : Iterable[int]
        A list of indices representing all join-irreducible concepts in the lattice.
    '''
    return list(reversed([node for node, child in lattice.children_dict.items() if len(child) == 1]))

def get_meet_irreducibles(lattice: ConceptLattice) -> Iterable[int]:
    '''
    Get all meet-irreducible concepts in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.

    Returns
    -------
    meet_irreducibles : Iterable[int]
        A list of indices representing all meet-irreducible concepts in the lattice.
    '''
    return [node for node, parent in lattice.parents_dict.items() if len(parent) == 1]

def is_distributive(lattice: ConceptLattice) -> bool:
    '''
    Check if the lattice is distributive.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.

    Returns
    -------
    distributive : bool
        True if the lattice is distributive, False otherwise.
    '''
    for (x, y, z) in combinations(lattice.to_networkx().nodes, 3):
        left = lattice.join([x, lattice.meet([y, z])])
        right = lattice.meet([lattice.join([x, y]), lattice.join([x, z])])
        if left != right:
            return False

        left = lattice.meet([x, lattice.join([y, z])])
        right = lattice.join([lattice.meet([x, y]), lattice.meet([x, z])])
        if left != right:
            return False

    return True

def is_join_distributive(lattice: ConceptLattice) -> bool:
    '''
    Check if the lattice is join-distributive.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.
    
    Returns
    -------
    join_distributive : bool
        True if the lattice is join-distributive, False otherwise.
    '''
    for x, y, z in combinations(list(lattice.to_networkx().nodes), 3):
        left = lattice.join([x, lattice.meet([y, z])])
        right = lattice.meet([lattice.join([x, y]), lattice.join([x, z])])
        if left != right:
            return False
    return True

def is_meet_distributive(lattice: ConceptLattice) -> bool:
    '''
    Check if the lattice is meet-distributive.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.

    Returns
    -------
    meet_distributive : bool
        True if the lattice is meet-distributive, False otherwise.
    '''
    for x, y, z in combinations(list(lattice.to_networkx().nodes), 3):
        left = lattice.meet([x, lattice.join([y, z])])
        right = lattice.join([lattice.meet([x, y]), lattice.meet([x, z])])
        if left != right:
            return False
    return True