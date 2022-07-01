from types import SimpleNamespace
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
from utils.preprocessing import clean_document_for_embedding
import json
import os


def compute_embeddings(path_to_files, embedding_model, output_path="./test_data/embeddings.json"):
    result = []
    for file in os.listdir(path_to_files):
        filename = os.fsdecode(file)
        if filename.endswith(".pdf"):
            preprocessed_document = clean_document_for_embedding(os.path.join(path_to_files, filename))
            sentence_embeddings = embedding_model.encode(preprocessed_document)
            result.append({"file_name": filename, "embeddings": sentence_embeddings.tolist()})
    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(result, f)
    return result


def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    model = args.model
    output_path = args.output_path
    input_path = args.input_path

    if isinstance(model, str):
        embedding_model = SentenceTransformer(model)
    else:
        embedding_model = model

    if isinstance(input_path, str):
        return compute_embeddings(input_path, embedding_model, output_path)
    else:
        raise ValueError("Invalid data type!")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--model",
                            type=str,
                            help="Embedding model path.",
                            default="./embedding_models/embedding_model")
    arg_parser.add_argument("--output_path",
                            type=str,
                            default="./test_data/embeddings.json",
                            help="If specified, write embeddings here. Currently assumed to be a json file.")
    arg_parser.add_argument("--input_path",
                            type=str,
                            help="Input data as a directory of pdf files.",
                            default="./test_data")
    parsed_args = arg_parser.parse_args()
    main(**vars(parsed_args))
    res_string = "Embeddings successfully computed"
    if parsed_args.output_path is not None:
        res_string += " and stored to {}".format(parsed_args.output_path)
    else:
        res_string += "."
    print(res_string)
