"""
Dissimilarity measures for clustering
"""

# Author: 'Nico de Vos' <njdevos@gmail.com>
# License: MIT

import numpy as np
import sklearn as sk  


def matching_dissim(a, b):
    """Simple matching dissimilarity function"""
               
    return sk.metrics.pairwise.pairwise_distances(np.atleast_2d(a),
                                                  np.atleast_2d(b), metric='l1', n_jobs=-1).astype(int).ravel()

def euclidean_dissim(a, b):
    """Euclidean distance dissimilarity function"""
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    return np.sum((a - b) ** 2, axis=1)
