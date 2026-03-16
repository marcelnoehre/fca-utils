# src/fca/context.py
import itertools
import pandas as pd
from typing import Set, Tuple

from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice


class Context:
    def __init__(self, formal_context: FormalContext):
        self.context = formal_context
        self.objects = formal_context.object_names
        self.attributes = formal_context.attribute_names
        self.G = set(self.objects)
        self.M = set(self.attributes)

    def object_concept(self, g: str) -> Tuple[Set[str], Set[str]]:
        '''
        Compute the object concept of g.

        Parameters
        ----------
        g : str
            The object

        Returns
        -------
        concept: Tuple[Set[str], Set[str]]
            The object concept
        '''
        extension = set(self.context.extension(intention))
        intention = set(self.context.intention({g}))
        return (extension, intention)

    def attribute_concept(self, m: str) -> Tuple[Set[str], Set[str]]:
        '''
        Compute the attribute concept of m.

        Parameters
        ----------
        m : str
            The attribute

        Returns
        -------
        concept: Tuple[Set[str], Set[str]]
            The attribute concept
        '''
        extension = set(self.context.extension({m}))
        intention = set(self.context.intention(extension))
        return (extension, intention)

    def object_closure(self, objects: Set[str]) -> Set[str]:
        '''
        Compute the closure of a set of objects.

        Parameters
        ----------
        objects : Set[str]
            The set of objects

        Returns
        -------
        closure: Set[str]
            The double-primed set of objects
        '''
        return set(self.context.extension(self.context.intention(objects)))

    def attribute_closure(self, attributes: Set[str]) -> Set[str]:
        '''
        Compute the closure of a set of attributes.

        Parameters
        ----------
        attributes : Set[str]
            The set of attributes

        Returns
        -------
        closure: Set[str]
            The double-primed set of attributes
        '''
        return set(self.context.intention(self.context.extension(attributes)))

    def reduce_context(self) -> "Context":
        '''
        Reduce the formal context by keeping only join-irreducible objects
        and meet-irreducible attributes.

        Returns
        -------
        reduced_context: Context
            The reduced formal context
        '''
        lattice = ConceptLattice.from_context(self.context)
        join_irreducibles = []
        meet_irreducibles = []

        for c in lattice.to_networkx().nodes:
            # join-irreducible: exactly one child
            if len(lattice.children(c)) == 1:
                join_irreducibles.append(list(lattice.get_concept_new_extent(c))[0])
            # meet-irreducible: exactly one parent
            if len(lattice.parents(c)) == 1:
                meet_irreducibles.append(list(lattice.get_concept_new_intent(c))[0])

        # already reduced
        if (
            len(join_irreducibles) == self.context.n_objects
            and len(meet_irreducibles) == self.context.n_attributes
        ):
            return self

        # build reduced context
        df = pd.DataFrame(0, index=join_irreducibles, columns=meet_irreducibles)
        for g, m in itertools.product(join_irreducibles, meet_irreducibles):
            if m in self.context.intention([g]):
                df.loc[g, m] = 1

        reduced_fc = FormalContext(
            data=df.values.astype(bool).tolist(),
            object_names=join_irreducibles,
            attribute_names=meet_irreducibles,
        )
        return Context(reduced_fc)
