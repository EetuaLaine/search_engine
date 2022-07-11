from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
from similar_documents import main as get_similar_documents
from sentence_transformers import SentenceTransformer


class Query(BaseModel):
    n: int
    query: str


app = FastAPI(
    title="Search engine API"
)


with open("./config.json") as f:
    config = json.load(f)


model = None

try:
    model = SentenceTransformer(config["embedding_model_path"])
    print("Successful model download.")
except Exception as e:
    print("Exception occurred when loading model:", e)


@app.get("/get_most_similar_documents", status_code=200)
def get_most_similar_documents(query: Query) -> dict:
    """
    Return n similar documents for a given query string.
    :param query: Query object.
    :return: Return document file names.
    """

    kwargs = query.dict()
    kwargs.update(config)

    return {"file_names": get_similar_documents(embedding_model=model, **kwargs)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
