import requests
from argparse import ArgumentParser
import json


def send_request(query, api_url, n=3):
    response = requests.get(api_url, json={"query": query, "n": n})
    return {"status": response.status_code, "results": json.loads(response.content)}


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--api_url", type=str, default="http:/0.0.0.0:8001/get_most_similar_documents")
    arg_parser.add_argument("--query", type=str)
    args = arg_parser.parse_args()
    print(send_request(args.query, args.api_url))
