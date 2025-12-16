import re
import copy
import subprocess

from typing import Iterable, Tuple
from fcapy.lattice import ConceptLattice

from src.fca_utils.parser import decode_cxt
from src.linear_extension.sat_realizer import SatRealizer
from src.fca_utils.lattice import *

def dim_draw_realizer(
        cxt_path: str,
        brunt_path: str
    ) -> Tuple[Iterable[int], Iterable[int]]:
    '''
    Compute the 2-dimensional 'realizer' using the conexp-clj API

    Parameters
    ----------
    cxt_path : str
        The path to a .cxt file
    brunt_path : str
        The path to the brunt standalone .jar

    Returns
    -------
    realizer : Tuple[Iterable[int], Iterable[int]]
        The 2-dimensional 'realizer' for DimDraw drawings
    '''
    context = decode_cxt(cxt_path)
    lattice = ConceptLattice.from_context(context)
    N = len(lattice.to_networkx().nodes())
    le_x, le_y = [None] * N, [None] * N

    # extent and intent for each node
    concepts = {}
    for node in lattice.to_networkx().nodes():
        concepts[node] = {
            g for _,g in extent_of_concept(lattice, node)
        }.union({
            m for _,m in intent_of_concept(lattice, node)
        })

    try:
        # execute conexp-clj
        res = subprocess.check_output([
            "java", "-jar", brunt_path, 
            '-f', 'dim-draw-coordinates',
            cxt_path], text=True)
        
        # derive realizer position from coordinates
        for line in res.splitlines():
            concept, coords = line.split(' -> ')
            x, y = coords.strip("()").split(", ")
            node = next((k for k, v in concepts.items() if v == set(re.findall(r"\b[g|m][A-Za-z0-9]+\b", concept))), None)
            le_x[int(x)] = node
            le_y[int(y)] = node

        return (le_x, le_y)

    except subprocess.CalledProcessError as e:
        print(f"Error running JAR: {e}")

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

    Notes
    -----
    Only suitable for small lattices due to combinatorial explosion.
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

def sat_realizer(lattice: ConceptLattice):
    '''
    Compute a minimal realizer using a SAT solver.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.

    Returns
    -------
    dimension : int
        The order dimension of the lattice (size of the minimal realizer).
    minimal_realizers : Iterable[Tuple[int]]
        A minimal realizer for the concept lattice.
    '''
    sat_realizer = SatRealizer(lattice)
    dim, sat_realizer = sat_realizer.realizer()
    realizer = [[list(lattice.to_networkx().nodes)[i] for i in le] for le in sat_realizer]
    
    return dim, realizer