from indexing import create_index
from utils.preprocessing import preprocess_document
from argparse import ArgumentParser


def most_similar_documents(query_string, compute_similarity, get_indices, n=3,
                           indices_path="./test_data/indices.json"):

    preprocessed_query = preprocess_document(query_string)
    query_index = create_index(preprocessed_query)

    indices = get_indices(indices_path=indices_path)

    similarities = [compute_similarity(index["index"], query_index) for index in indices]

    zipped = sorted([(idx, sim) for idx, sim in enumerate(similarities)], key=lambda t: -t[1])

    indices_to_return = [t[0] for t in zipped[:n]]

    return [indices[idx]["file_name"] for idx in indices_to_return]


# TODO: implement this so that it calls most_similar_documents with functions determined by kwargs,
#  put the functions as values in a dictionary where keys are names.
def main(**kwargs):
    pass


if __name__ == "__main__":
    arg_parser = ArgumentParser()
