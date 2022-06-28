import os
import json
from argparse import ArgumentParser
import pandas as pd

from utils.preprocessing import preprocess_pdf_document


def create_index(preprocessed_document):
    return pd.Series(preprocessed_document).value_counts().to_dict()


def index_pdf_document(input_pdf):
    preprocessed_file = preprocess_pdf_document(input_pdf)
    return create_index(preprocessed_file)


# TODO: Match output path to some kind of regex and depending on its form, write to a DB, S3 or local filesystem.
def compute_indices(path_to_files: str, output_path: str = "./test_data/indices.json") -> None:
    """
    Iterate over directory, index all the pdf documents in the directory (non-recursively), write out to json.
    :param path_to_files: Path to a directory containing pdf documents.
    :param output_path: Path to output json file.
    :return: Nothing.
    """
    result = []
    for file in os.listdir(path_to_files):
        filename = os.fsdecode(file)
        if filename.endswith(".pdf"):
            index = index_pdf_document(os.path.join(path_to_files, filename))
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


# TODO: Take in arguments and perform either indexing or index retrieval with specified methods.
#  Store functions in dict.
def main(**kwargs):
    pass


if __name__ == "__main__":
    arg_parser = ArgumentParser()
