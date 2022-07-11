# TODO: Get a base image with pytorch preinstalled
FROM python:3.8-slim-bullseye

WORKDIR /app

ADD ./requirements.txt /app/requirements.txt

RUN python -m pip install nltk && \
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')" && \
    python -m pip install -r requirements.txt && \
    python -m pip install pdfminer

ADD . /app

EXPOSE 8001

CMD ["python", "search_engine_api.py"]