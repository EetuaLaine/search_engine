from types import SimpleNamespace
from utils.similarity_functions import *
from indexing import create_index, read_indices_from_json
from utils.preprocessing import preprocess_document
from argparse import ArgumentParser


def most_similar_documents(query_string, compute_similarity, get_indices, n=3,
                           indices_path="./test_data/indices.json"):

    preprocessed_query = preprocess_document(query_string)
    query_index = create_index(preprocessed_query)

    indices = get_indices(indices_path=indices_path)

    # TODO: Pass the data to compute_similarity as *args or **kwargs so you can use vector embeddings
    #  as well as the indices depending on the similarity function
    similarities = [compute_similarity(index["index"], query_index) for index in indices]

    zipped = sorted([(idx, sim) for idx, sim in enumerate(similarities)], key=lambda t: -t[1])

    indices_to_return = [t[0] for t in zipped[:n]]

    return [indices[idx]["file_name"] for idx in indices_to_return]


# TODO: implement this so that it calls most_similar_documents with functions determined by kwargs,
#  put the functions as values in a dictionary where keys are names.
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    similarity_functions = {"lexical": index_similarity}
    index_getter_functions = {"json": read_indices_from_json}
    get_similarity = similarity_functions[args.similarity_func]
    query_string = args.query
    index_source = args.index_source


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--similarity_func",
                            type=str,
                            default="lexical",
                            choices=["lexical"],
                            help="How to compute similarity between sentences and documents")
    arg_parser.add_argument("--query",
                            type=str,
                            help="This parameter contains the search query.")
    arg_parser.add_argument("--index_source",
                            type=str,
                            choices=["json"],
                            help="Determines the method of obtaining the indices.")
    arg_parser.add_argument("--index_path",
                            type=str,
                            help="Path to the indices. If the index source is json, this should be a local filepath.")

    parsed_args = arg_parser.parse_args()
    main(**vars(parsed_args))
