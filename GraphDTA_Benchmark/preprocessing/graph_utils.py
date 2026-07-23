"""
Utilities for converting SMILES strings into molecular graphs.

Each molecule is represented as:
    - Number of atoms
    - Node (atom) features
    - Edge indices
"""

import numpy as np
import networkx as nx
from rdkit import Chem


def one_of_k_encoding(x, allowable_set):
    """
    One-hot encoding.
    Raises an exception if x is not in the allowable set.
    """
    if x not in allowable_set:
        raise ValueError(f"{x} not in allowable set {allowable_set}")

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """
    Unknown values are mapped to the last element.
    """
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    """
    Generate atom feature vector.
    """

    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
                'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
                'Unknown'
            ]
        )
        +
        one_of_k_encoding(atom.GetDegree(),
                          list(range(11)))
        +
        one_of_k_encoding_unk(atom.GetTotalNumHs(),
                              list(range(11)))
        +
        one_of_k_encoding_unk(atom.GetImplicitValence(),
                              list(range(11)))
        +
        [atom.GetIsAromatic()],
        dtype=np.float32
    )


def smile_to_graph(smile):
    """
    Convert a SMILES string into a graph.

    Returns
    -------
    num_atoms : int

    node_features : List[np.ndarray]

    edge_index : List[List[int]]
    """

    mol = Chem.MolFromSmiles(smile)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")

    num_atoms = mol.GetNumAtoms()

    node_features = []

    for atom in mol.GetAtoms():

        feature = atom_features(atom)

        if feature.sum() != 0:
            feature /= feature.sum()

        node_features.append(feature)

    edges = []

    for bond in mol.GetBonds():

        edges.append([
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx()
        ])

    graph = nx.Graph(edges).to_directed()

    edge_index = []

    for src, dst in graph.edges():
        edge_index.append([src, dst])

    return num_atoms, node_features, edge_index