from send_single_request_to_api import send_request
import json


API_URL = "http://0.0.0.0:8001/"
CMD = "get_most_similar_documents"


def test_success():
    query = "Rarest fish in Greenland?!"
    response = send_request(query, API_URL + CMD)
    assert "status" in response and \
           response["status"] == 200 and \
           "results" in response and \
           "file_names" in response["results"] and \
           isinstance(response["results"]["file_names"], list), \
           "Invalid output from API!"


def test_corpus():
    with open("./test_data/corpus_tests.json") as f:
        corpus_tests = json.load(f)

    success_rate = 0

    for test in corpus_tests:
        query = test["query"]
        response = send_request(query, API_URL + CMD)
        file_names = response["results"]["file_names"]
        if test["answer"] in file_names:
            success_rate += 1

    success_rate /= len(corpus_tests)

    assert success_rate >= 0.8, "Success rate is only {}".format(success_rate)
