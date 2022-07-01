from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer


def load_embedding_model(src_dir, target_dir):
    model = SentenceTransformer(src_dir)
    model.save(target_dir)
    print("Model loaded from {} and successfully saved to {}.".format(src_dir, target_dir))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--src_dir",
                            type=str,
                            help="""The directory from which to load model. 
                            Typically this would point to the Huggingface model repository.""",
                            default="paraphrase-MiniLM-L6-v2")
    arg_parser.add_argument("--target_dir",
                            type=str,
                            help="""The directory to save the loaded model. 
                            Typically this would point to a local directory.""",
                            default="./embedding_models/embedding_model")
    args = arg_parser.parse_args()

    load_embedding_model(args.src_dir, args.target_dir)
