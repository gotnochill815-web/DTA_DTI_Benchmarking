"""
Utilities for encoding protein sequences into fixed-length integer vectors.
"""

import numpy as np

# Protein vocabulary (same as original GraphDTA)
SEQ_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

# Map amino acid -> integer
SEQ_DICT = {char: idx + 1 for idx, char in enumerate(SEQ_VOCAB)}

# Maximum protein sequence length
MAX_SEQ_LEN = 1000


def encode_protein(sequence, max_length=MAX_SEQ_LEN):
    """
    Encode a protein sequence into a fixed-length integer array.

    Parameters
    ----------
    sequence : str
        Protein amino acid sequence.

    max_length : int
        Maximum sequence length.

    Returns
    -------
    np.ndarray
        Integer encoded protein sequence.
    """

    encoded = np.zeros(max_length, dtype=np.int64)

    for i, amino_acid in enumerate(sequence[:max_length]):
        encoded[i] = SEQ_DICT.get(amino_acid, 0)

    return encoded