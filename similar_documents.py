from types import SimpleNamespace

from sentence_transformers import SentenceTransformer

from utils.similarity_functions import *
from indexing import create_index, read_indices_from_json
from utils.preprocessing import preprocess_document
from argparse import ArgumentParser
from sentence_embedding import embed_query, get_embeddings


def most_similar_documents(query_string, compute_similarity, get_indices,
                           n=3, get_embeddings_func=None, embedding_function=None,
                           embedding_model=None, indices_path="./test_data/indices.json",
                           embeddings_path=None):

    preprocessed_query = preprocess_document(query_string)
    query_index = create_index(preprocessed_query)

    indices = get_indices(indices_path=indices_path)

    if get_embeddings_func is None or embedding_function is None or embedding_model is None:
        similarities = [compute_similarity(index["index"], query_index) for index in indices]
    else:
        embeddings = get_embeddings_func(embeddings_path)
        embedded_query = embedding_function(query_string, embedding_model)
        similarities = [compute_similarity(index["index"], query_index,
                                           document_embedding=embeddings[i],
                                           sentence_embedding=embedded_query) for i, index in enumerate(indices)]

    zipped = sorted([(idx, sim) for idx, sim in enumerate(similarities)], key=lambda t: -t[1])

    indices_to_return = [t[0] for t in zipped[:n]]

    return [indices[idx]["file_name"] for idx in indices_to_return]


def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    similarity_functions = {"lexical": index_similarity, "combined": combined_similarity}
    index_getter_functions = {"json": read_indices_from_json}
    embeddings_loading_functions = {"json": get_embeddings}
    embedding_functions = {"sentence-transformers": embed_query}
    query_string = args.query
    index_source = args.index_source
    embeddings_source = args.embeddings_source
    embedding_function_type = args.embedding_function_type
    similarity_func = args.similarity_func
    get_similarity = similarity_functions[similarity_func]
    get_indices = index_getter_functions[index_source]
    n = args.n
    index_path = args.index_path
    embeddings_path = args.embeddings_path
    get_embeddings_func = embeddings_loading_functions[embeddings_source]
    embedding_function = embedding_functions[embedding_function_type]
    embedding_model_path = args.embedding_model_path

    embedding_model = None

    if embedding_model_path is not None:
        embedding_model = SentenceTransformer(embedding_model_path)

    return most_similar_documents(query_string, get_similarity,
                                  get_indices, n=n, indices_path=index_path,
                                  get_embeddings_func=get_embeddings_func, embedding_function=embedding_function,
                                  embedding_model=embedding_model, embeddings_path=embeddings_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--similarity_func",
                            type=str,
                            default="lexical",
                            choices=["lexical", "combined"],
                            help="How to compute similarity between sentences and documents")
    arg_parser.add_argument("--query",
                            type=str,
                            help="This parameter contains the search query.")
    arg_parser.add_argument("--index_source",
                            type=str,
                            choices=["json"],
                            help="Determines the method of obtaining the indices.")
    arg_parser.add_argument("--embeddings_source",
                            type=str,
                            choices=["json"],
                            help="Determines the method of obtaining the embeddings.")
    arg_parser.add_argument("--index_path",
                            type=str,
                            help="Path to the indices. If the index source is json, this should be a local filepath.")
    arg_parser.add_argument("--embeddings_path",
                            type=str,
                            help="Path to embeddings. Should be a local json filepath.")
    arg_parser.add_argument("--embedding_model_path",
                            type=str,
                            help="Path to load embedding model. In the repository by default.",
                            default="./embedding_models/embedding_model")
    arg_parser.add_argument("--n",
                            type=int,
                            default=3,
                            help="How many documents to return at most.")
    arg_parser.add_argument("--embedding_function_type",
                            type=str,
                            help="What kind of function to use for embedding sentences.",
                            choices=["sentence-transformers"],
                            default="sentence-transformers")

    parsed_args = arg_parser.parse_args()
    print(main(**vars(parsed_args)))
