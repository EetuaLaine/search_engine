from types import SimpleNamespace
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser


def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    model = args.model

    if isinstance(model, str):
        # Treat as a model path
        pass
    else:
        # Treat as a model object
        pass


if __name__ == "__main__":
    arg_parser = ArgumentParser()
