import numpy as np
from numpy.linalg import norm


def cosine_similarity(x1, x2, **kwargs):
    return np.dot(x1, x2) / (norm(x1) * norm(x2))


def index_similarity(idx1, idx2, **kwargs):
    sum_1 = sum(idx1.values())
    sum_2 = sum(idx2.values())

    intersection = 0

    if sum_1 < sum_2:
        min_idx = idx1
        max_idx = idx2
        min_sum = sum_1
    else:
        min_idx = idx2
        max_idx = idx1
        min_sum = sum_2

    for k, v in min_idx.items():
        intersection += min(max_idx.get(k, 0), v)

    return intersection / min_sum
