import os
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict
from itertools import combinations
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

from fca.lattice import Lattice

def plot_lattice(
        context: FormalContext,
        concepts: List[int],
        coordinates : Dict,
        title: str = '', 
        annotations: bool = False, 
        origin: bool = False
    ):
    '''
    Plot the Concept Lattice.

    Parameters
    ----------
    context : FormalContext
        The formal context
    concepts : List[int]
        The list of all formal concepts
    coordinates : Dict
        The positions of the formal concepts in the drawing plane
    title : str
        The title for the plot window
    annotations : bool
        Whether to display object and attribute labels
    origin : bool
        Whether to display the origin
    '''
    lattice = ConceptLattice.from_context(context)
    fig = plt.figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title(title)

    if origin:
        plt.scatter(0, 0, facecolor='red', edgecolor='pink', linewidth=5, s=150, zorder=10)
    
    # vertices
    for concept in concepts:
        (x, y) = coordinates[concept]
        plt.scatter(x, y, facecolor='white', edgecolor='black', linewidth=2.5, s=150, zorder=4)
        if annotations:
            # attributes
            plt.annotate(
                ','.join(lattice.get_concept_new_intent(concept)),
                (x, y),
                textcoords='offset points',
                xytext=(0, 10),
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
            # objects
            plt.annotate(
                ','.join(lattice.get_concept_new_extent(concept)),
                (x, y),
                textcoords='offset points',
                xytext=(0, -10),
                ha='center',
                va='top',
                fontsize=12,
                fontweight='bold'
            )

    # edges
    for (i, j) in Lattice(context).cover_relations():
        x_0, y_0 = np.array(coordinates[i])
        x_1, y_1 = np.array(coordinates[j])
        plt.plot([x_0, x_1], [y_0, y_1], color='black', linewidth=2.5, zorder=2)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_si_graph(
        d_si_points: List, 
        elements: List[str], 
        n_1: int, 
        n_2: int, 
        annotations: bool
    ):
    '''
    Plot the Supremum-Infimum graph representing the relationship 
    between objects and attributes in the layout space.

    Parameters
    ----------
    d_si_points : List
        The nodes of the d_SI graph
    elements: List[str]
        The element labels for the graph
    n_1 : int
        The node on the left
    n_2 : int
        The node on the right
    annotations : bool
        Whether to display object and attribute labels
    '''
    fig = plt.figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title(f'Sup Inf Graph: {vars.cxt}')

    # elements as vertices
    for i, (x, y) in enumerate(d_si_points):
        plt.scatter(x, y, facecolor='white', edgecolor='black', linewidth=2.5, s=150, zorder=4)
        if annotations:
            plt.annotate(
                elements[i],
                (x, y),
                textcoords='offset points',
                xytext=(0, 10),
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

    # edges approximate d_SI
    for i, j in combinations(range(len(elements)), 2):
        x_0, y_0 = np.array(d_si_points[i])
        x_1, y_1 = np.array(d_si_points[j])
        plt.plot([x_0, x_1], [y_0, y_1], color='black', linewidth=1, zorder=2)

    # f_max
    plt.plot([n_1[0], n_2[0]], [n_1[1], n_2[1]], color='black', linewidth=4, zorder=2)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def tikz_export(
        context: FormalContext,
        concepts: List[int],
        coordinates : Dict,
        path: str
    ):
    '''
    Generate a LaTeX PGF/TikZ representation of the concept lattice.

    Parameters
    ----------
    context : FormalContext
        The formal context
    concepts : List[int]
        The list of all formal concepts
    coordinates : Dict
        The positions of the formal concepts in the drawing plane
    path : str
        The path to store the figure
    '''
    lines = [
        r'\begin{tikzpicture}[scale=1.0]',
        r'  \begin{scope}[every node/.style={circle, thick, draw, fill=white, inner sep=0pt, minimum size=2mm}]'
    ]

    # vertices
    for c in concepts:
        (x, y) = coordinates[c]
        lines.append(fr'    \node ({c}) at ({x:.3f}, {y:.3f}) {{}};')

    lines.append(r'  \end{scope}')

    # edges
    for (i, j) in Lattice(context).cover_relations():
        lines.append(fr'  \draw ({i}) -- ({j});')
        
    lines.append(r'\end{tikzpicture}')

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def pdf_export(
        context: FormalContext,
        concepts: List[int],
        coordinates : Dict,
        path: str
    ):
    '''
    Export the Concept Lattice as a PDF file.

    Parameters
    ----------
    context : FormalContext
        The formal context
    concepts : List[int]
        The list of all formal concepts
    coordinates : Dict
        The positions of the formal concepts in the drawing plane
    path : str
        The path to store the figure
    '''
    plt.figure(figsize=(6, 6))

    # vertices
    for concept in concepts:
        (x, y) = coordinates[concept]
        plt.scatter(x, y, facecolor='white', edgecolor='black', linewidth=3.5, s=350, zorder=4)

    # edges
    for (i, j) in Lattice(context).cover_relations():
        x_0, y_0 = np.array(coordinates[i])
        x_1, y_1 = np.array(coordinates[j])
        plt.plot([x_0, x_1], [y_0, y_1], color='black', linewidth=3.5, zorder=2)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.close()
