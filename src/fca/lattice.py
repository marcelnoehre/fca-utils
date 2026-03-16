# src/fca/lattice.py
import networkx as nx

from collections import deque
from itertools import combinations
from typing import Set, Tuple, List, Iterable, Dict

from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

class Lattice:
    def __init__(self, context: FormalContext):
        self.lattice = ConceptLattice.from_context(context)
        self.nodes = self.lattice.to_networkx().nodes()

    def concepts(self) -> List[Tuple[Set[str], Set[str]]]:
        '''
        Return all concepts in the lattice.

        Returns
        -------
        concepts: List[Tuple[Set[str], Set[str]]]
            List of (extent, intent) pairs
        '''
        return [
            (
                set(self.lattice.get_concept_new_extent(c)),
                set(self.lattice.get_concept_new_intent(c)),
            )
            for c in self.lattice.to_networkx().nodes
        ]

    def children(self, concept: int) -> List[int]:
        '''
        Return direct child concept indices of a given concept.

        Parameters
        ----------
        concept : int
            The index of the formal concept.

        Returns
        -------
        children : List[int]
            The list of all children indices. 
        '''
        return self.lattice.children(concept)

    def parents(self, concept: int) -> List[int]:
        '''
        Return direct parent concept indices of a given concept.
        
        Parameters
        ----------
        concept : int
            The index of the formal concept.
        
        Returns
        -------
        parents : List[int]
            The list of all parent indices.
        '''
        return self.lattice.parents(concept)

    def all_children(self, index: int) -> Set[int]:
        '''
        Get all children of a concept in the lattice (transitive).

        Parameters
        ----------
        index : int
            The index of the concept.

        Returns
        -------
        children : Set[int]
        '''
        visited = set()
        queue = deque([index])
        children = set()

        while queue:
            node = queue.popleft()
            for child in self.lattice.children(node):
                if child not in visited:
                    visited.add(child)
                    children.add(child)
                    queue.append(child)

        return children

    def all_parents(self, index: int) -> Set[int]:
        '''
        Get all parents of a concept in the lattice (transitive).

        Parameters
        ----------
        index : int
            The index of the concept.

        Returns
        -------
        parents : Set[int]
        '''
        visited = set()
        queue = deque([index])
        parents = set()

        while queue:
            node = queue.popleft()
            for parent in self.lattice.parents(node):
                if parent not in visited:
                    visited.add(parent)
                    parents.add(parent)
                    queue.append(parent)

        return parents
    
    def all_extents(self) -> Dict[int, Set[str]]:
        '''
        Compute the extents for all concepts in the lattice.

        Returns
        -------
        extents: Dict[int, Set[str]]
            A dictionary mapping concept IDs to their full extents
        '''
        seen = set({})
        queue = deque({len(self.lattice.to_networkx().nodes)-1})
        extents = dict({})

        while queue:
            concept = queue.popleft()
            
            # all childrens processed?
            if self.lattice.children(concept) <= extents.keys():
                # A_new \cup A_children
                extents[concept] = self.lattice.get_concept_new_extent(concept).union(
                    *(extents[c] for c in self.lattice.children(concept))
                )

                # add parents to queue
                seen.update(self.lattice.parents(concept) - extents.keys())
                queue.extend(self.lattice.parents(concept) - extents.keys())

            # readd to queue
            else:
                queue.append(concept)

        return extents

    def all_intents(self) -> Dict[int, Set[str]]:
        '''
        Compute the intents for all concepts in the lattice.

        Returns
        -------
        intents: Dict[int, Set[str]]
            A dictionary mapping concept IDs to their full intents
        '''
        seen = set({})
        queue = deque({0})
        intents = dict({})

        while queue:
            concept = queue.popleft()
            
            # all parents processed?
            if self.lattice.parents(concept) <= intents.keys():
                # B_new \cup B_parents
                intents[concept] = self.lattice.get_concept_new_intent(concept).union(
                    *(intents[p] for p in self.lattice.parents(concept))
                )

                # add children to queue
                seen.update(self.lattice.children(concept) - intents.keys())
                queue.extend(self.lattice.children(concept) - intents.keys())

            # readd to queue
            else:
                queue.append(concept)

        return intents

    def intent_of_concept(self, index: int) -> Set[Tuple[int, str]]:
        '''
        Get the full intent of a concept (including inherited from parents).

        Parameters
        ----------
        index : int
            The index of the concept.

        Returns
        -------
        intent : Set[Tuple[int, str]]
        '''
        parents = self.all_parents(index)
        intent = {(index, feature) for feature in self.lattice.get_concept_new_intent(index)}
        for parent in parents:
            for feature in self.lattice.get_concept_new_intent(parent):
                intent.add((parent, feature))
        return intent

    def extent_of_concept(self, index: int) -> Set[Tuple[int, str]]:
        '''
        Get the full extent of a concept (including inherited from children).

        Parameters
        ----------
        index : int
            The index of the concept.

        Returns
        -------
        extent : Set[Tuple[int, str]]
        '''
        children = self.all_children(index)
        extent = {(index, obj) for obj in self.lattice.get_concept_new_extent(index)}
        for child in children:
            for obj in self.lattice.get_concept_new_extent(child):
                extent.add((child, obj))
        return extent

    def join_irreducibles(self) -> Iterable[int]:
        '''
        Get all join-irreducible concept indices (exactly one child).

        Returns
        -------
        join_irreducibles : Iterable[int]
        '''
        return list(
            reversed([node for node, child in self.lattice.children_dict.items() if len(child) == 1])
        )

    def meet_irreducibles(self) -> Iterable[int]:
        '''
        Get all meet-irreducible concept indices (exactly one parent).

        Returns
        -------
        meet_irreducibles : Iterable[int]
        '''
        return [node for node, parent in self.lattice.parents_dict.items() if len(parent) == 1]

    def cover_relations(self) -> Set[Tuple[int, int]]:
        '''
        Get the cover relations of a concept lattice.

        Parameters
        ----------
        concept_lattice : ConceptLattice
            The concept lattice.

        Returns
        -------
        cover_relations : Set[Tuple[int, int]]
            A set of tuples representing the cover relations of the lattice.
        '''
        return set(nx.transitive_reduction(self.lattice.to_networkx()).edges)

    def transitive_closure(self) -> Set[Tuple[int, int]]:
        '''
        Get the transitive closure of the lattice.

        Returns
        -------
        transitive_closure : Set[Tuple[int, int]]
        '''
        return set(nx.transitive_closure(self.lattice.to_networkx()).edges)

    def incomparability_graph(self) -> nx.Graph:
        """
        Get the incomparability graph of the lattice.

        Returns
        -------
        incomparability_graph : nx.Graph
        """
        return nx.complement(nx.transitive_closure(self.lattice.to_networkx()).to_undirected())

    def is_distributive(self) -> bool:
        """
        Check if the lattice is distributive.

        Returns
        -------
        bool
        """
        for x, y, z in combinations(self.lattice.to_networkx().nodes, 3):
            if self.lattice.join([x, self.lattice.meet([y, z])]) != self.lattice.meet([self.lattice.join([x, y]), self.lattice.join([x, z])]):
                return False
            if self.lattice.meet([x, self.lattice.join([y, z])]) != self.lattice.join([self.lattice.meet([x, y]), self.lattice.meet([x, z])]):
                return False
        return True

    def is_join_distributive(self) -> bool:
        """
        Check if the lattice is join-distributive.

        Returns
        -------
        bool
        """
        for x, y, z in combinations(self.lattice.to_networkx().nodes, 3):
            if self.lattice.join([x, self.lattice.meet([y, z])]) != self.lattice.meet([self.lattice.join([x, y]), self.lattice.join([x, z])]):
                return False
        return True

    def is_meet_distributive(self) -> bool:
        """
        Check if the lattice is meet-distributive.

        Returns
        -------
        bool
        """
        for x, y, z in combinations(self.lattice.to_networkx().nodes, 3):
            if self.lattice.meet([x, self.lattice.join([y, z])]) != self.lattice.join([self.lattice.meet([x, y]), self.lattice.meet([x, z])]):
                return False
        return True