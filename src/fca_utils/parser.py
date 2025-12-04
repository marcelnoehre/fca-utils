import math

from typing import Iterable, Tuple

def decode_lce(lce: str) -> Iterable[Tuple[int, int]]:
    '''
    Decode a levellised covering encoding (LCE) string into a list of covering pairs.
    
    Each character corresponds to an entry in the upper-triangular part of the incidence matrix.

    '1' indicates the a covering pair (a, b), while any other character indicates its absence.

    Parameters
    ----------
    lce : str
        A string representing the levellised covering encoding.

    Returns
    -------
    covering_pairs : Iterable[Tuple[int, int]]
        An iterable of tuples representing the covering pairs.
    '''
    return [pair
        for i, pair in enumerate(([(a, b) 
            for b in range(1, int(0.5 + math.sqrt(0.25 + 2 * len(lce)))) 
            for a in range(b)])) 
        if lce[i] == '1'
    ]
