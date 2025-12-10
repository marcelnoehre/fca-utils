import re
import subprocess

from typing import Iterable, Tuple

from src.fca_utils.parser import decode_cxt
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