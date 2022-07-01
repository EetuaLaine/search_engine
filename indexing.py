import os
import json
from argparse import ArgumentParser
from types import SimpleNamespace
import pandas as pd
from utils.preprocessing import preprocess_pdf_document
from tqdm import tqdm


def create_index(preprocessed_document):
    return pd.Series(preprocessed_document).value_counts().to_dict()


def index_pdf_document(input_pdf):
    preprocessed_file = preprocess_pdf_document(input_pdf)
    return create_index(preprocessed_file)


# TODO: Match output path to some kind of regex and depending on its form, write to a DB, S3 or local filesystem.
#  TODO: This method overwrites all existing indices!
def compute_indices(path_to_files: str,
                    index_storing_format: str = "json",
                    output_path: str = "./test_data/indices.json") -> None:
    """
    Iterate over directory, index all the pdf documents in the directory (non-recursively), write out to json.
    :param index_storing_format: Format to store the indices. Currently only supports json.
    :param path_to_files: Path to a directory containing pdf documents.
    :param output_path: Path to output json file.
    :return: Nothing.
    """
    if index_storing_format != "json":
        raise ValueError("Only json format is currently supported for storing indices!")

    result = []
    for file in tqdm(os.listdir(path_to_files)):
        filename = os.fsdecode(file)
        if filename.endswith(".pdf"):
            preprocessed_document = preprocess_pdf_document(os.path.join(path_to_files, filename))
            index = create_index(preprocessed_document)
            result.append({"file_name": filename, "index": index})
    with open(output_path, "w") as f:
        json.dump(result, f)


def read_indices_from_json(indices_path=None):
    try:
        with open(indices_path) as f:
            return json.load(f)
    except Exception as e:
        print("JSON loading failed with exception:", e)
        return None


def main(**kwargs):
    indexing_functions = {"bag_of_words": compute_indices}
    index_reading_functions = {"json": read_indices_from_json}
    # kwargs into namespace
    args = SimpleNamespace(**kwargs)
    # Get parameters from args.
    action = args.action
    if action == "write_indices":
        indexing_method = args.indexing_method
        pdf_document_directory = args.pdf_document_directory
        indexing_function = indexing_functions[indexing_method]
    index_path = args.index_path
    index_storing_format = args.index_storing_format
    # Choose indexing functions.
    index_reading_function = index_reading_functions[index_storing_format]

    if action == "read_indices":
        return index_reading_function(index_path)
    elif action == "write_indices":
        indexing_function(pdf_document_directory, index_storing_format)
        return "Indices successfully computed and written to file."
    else:
        raise ValueError("action has invalid value")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--action",
                            type=str,
                            choices=["read_indices", "write_indices"],
                            help="Determines whether to obtain indices or to index new document(s).")
    arg_parser.add_argument("--indexing_method",
                            type=str,
                            choices=["bag_of_words"],
                            help="Method to compute indices. Ignored if action=read_indices.")
    arg_parser.add_argument("--index_storing_format",
                            type=str,
                            choices=["json"],
                            help="Format to store indices. Ignored if action=write_indices.")
    arg_parser.add_argument("--index_path",
                            type=str,
                            help="Path to fetch indices, or write indices. Format depends on index_storing_format.")
    arg_parser.add_argument("--pdf_document_directory",
                            type=str,
                            help="Directory containing the pdf documents to index. Ignored if action=read_indices.")

    parsed_args = arg_parser.parse_args()

    print(main(**vars(parsed_args)))
