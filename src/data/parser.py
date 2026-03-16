# src/data/parser.py
import math
import re
import pandas as pd
import networkx as nx

from typing import Iterable, Tuple
from fcapy.context import FormalContext

class Parser:

    def decode_lce(self, lce: str) -> FormalContext:
        '''
        Decode a levellised covering encoding (LCE) string into a Formal Context.
        
        Each character corresponds to an entry in the upper-triangular part of the incidence matrix.

        '1' indicates the presence of a covering pair (a, b), while any other character indicates its
        absence.

        Parameters
        ----------
        lce : str
            A string representing the levellised covering encoding or a path to the .lce file

        Returns
        -------
        formal_context : FormalContext
            The formal context.
        '''
        if lce.endswith('.lce'):
            with open(lce, 'r') as f:
                lce = f.read()

        return self.from_covers([pair
            for i, pair in enumerate(([(a, b) 
                for b in range(1, int(0.5 + math.sqrt(0.25 + 2 * len(lce)))) 
                for a in range(b)])) 
            if lce[i] == '1'
        ], int((1 + math.sqrt(1 + 8 * len(lce))) / 2))

    def decode_cxt(self, cxt: str) -> FormalContext:
        '''
        Decode a Burmeister (B) string into a Formal Context.

        The string starts with a B, followed by the dimension of the context and the incidence matrix.

        'x' or 'X' indicates that a object (row) has a feature (column), while a any other character
        indicates that a object does not have a feature. 

        Parameters
        ----------
        cxt : str
            A string representing the burmeister format or a path to the .cxt file

        Returns
        -------
        formal_context : FormalContext
            The formal context.
        '''
        if cxt.endswith('.cxt'):
            with open(cxt, 'r') as f:
                cxt = f.read()

        _, ns, cxt = cxt.split('\n\n')
        n_objs, n_attrs = [int(x) for x in ns.split('\n')]

        cxt = cxt.strip().split('\n')
        obj_names, cxt = cxt[:n_objs], cxt[n_objs:]
        attr_names, cxt = cxt[:n_attrs], cxt[n_attrs:]
        cxt = [[(c == 'X' or c == 'x') for c in line] for line in cxt]

        return FormalContext(data=cxt, object_names=obj_names, attribute_names=attr_names)

    def decode_conexp_simple(self, cxt: str):
        '''
        Decode a context file exported from ConExp-Clj.

        Parameters
        ----------
        cxt : str
            A string from ConExp-Clj or a path to the .cxt file

        Returns
        -------
        formal_context : FormalContext
            The formal context.
        '''
        if cxt.endswith('.cxt'):
            with open(cxt, 'r') as f:
                cxt = f.read()
                
        G_str, M_str, I_str = cxt.split('\n')[1].split('#')[1:]
        G = re.findall(r'"(.*?)"', G_str)
        M = re.findall(r'"(.*?)"', M_str)
        I = re.findall(r'\["(.*?)"\s+"(.*?)"\]', I_str)

        df = pd.DataFrame(0, index=G, columns=M)
        for g, m in I:
            if g in df.index and m in df.columns:
                df.loc[g, m] = 1

        return FormalContext(data=df.values.astype(bool).tolist(), object_names=G, attribute_names=M)

    def from_covers(self, covers: Iterable[Tuple[int, int]], N: int) -> FormalContext:
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


