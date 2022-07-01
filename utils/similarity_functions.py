import numpy as np
from numpy.linalg import norm


# TODO: Generalize for matrices.
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


def semantic_similarity(document_embeddings, query_embedding, k=5, **kwargs):
    similarities = [cosine_similarity(doc_sent, query_embedding) for doc_sent in document_embeddings]
    return sum(sorted(similarities)[-k:]) / k


# TODO: This function needs to take all the parameters required by index similarity and
#  semantic similarity, compute them both and return some kind of weighted average.
#  Consider scaling the cosine similarity to [0, 1].
def voted_similarity(**kwargs):
    pass
