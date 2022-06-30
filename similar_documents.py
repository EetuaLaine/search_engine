from types import SimpleNamespace
from utils.similarity_functions import *
from indexing import create_index, read_indices_from_json
from utils.preprocessing import preprocess_document
from argparse import ArgumentParser


def most_similar_documents(query_string, compute_similarity, get_indices,
                           n=3, get_embeddings=None, embedding_function=None,
                           indices_path="./test_data/indices.json"):

    preprocessed_query = preprocess_document(query_string)
    query_index = create_index(preprocessed_query)

    indices = get_indices(indices_path=indices_path)

    if get_embeddings is None or embedding_function is None:
        similarities = [compute_similarity(index["index"], query_index) for index in indices]
    else:
        embeddings = get_embeddings()
        embedded_query = embedding_function(query_string)
        similarities = [compute_similarity(index["index"], query_index,
                                           document_embedding=embeddings[i],
                                           sentence_embedding=embedded_query) for i, index in enumerate(indices)]

    zipped = sorted([(idx, sim) for idx, sim in enumerate(similarities)], key=lambda t: -t[1])

    indices_to_return = [t[0] for t in zipped[:n]]

    return [indices[idx]["file_name"] for idx in indices_to_return]


def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    similarity_functions = {"lexical": index_similarity}
    index_getter_functions = {"json": read_indices_from_json}
    query_string = args.query
    index_source = args.index_source
    similarity_func = args.similarity_func
    get_similarity = similarity_functions[similarity_func]
    get_indices = index_getter_functions[index_source]
    n = args.n
    index_path = args.index_path
    return most_similar_documents(query_string, get_similarity, get_indices, n=n, indices_path=index_path)


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
    arg_parser.add_argument("--n",
                            type=int,
                            default=3,
                            help="How many documents to return at most.")

    parsed_args = arg_parser.parse_args()
    print(main(**vars(parsed_args)))
