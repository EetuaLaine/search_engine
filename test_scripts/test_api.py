from send_single_request_to_api import send_request


API_URL = "http://0.0.0.0:8001/"


def test_success():
    cmd = "get_most_similar_documents"
    query = "Rarest fish in Greenland?!"
    response = send_request(query, API_URL + cmd)
    assert "status" in response and response["status"] == 200
