# Search engine project

## Supported usage

### Index a directory of pdf documents and store results into json.

```console
$ python3 indexing.py --action write_indices --indexing_method bag_of_words --index_storing_format json --index_path ./test_data/indices.json --pdf_document_directory test_data
```

### Read indices from a json file.

```console
$ python3 indexing.py --action read_indices --index_storing_format json --index_path ./test_data/indices.json
```

### Get 3 most similar documents to a query using just the indices obtained from a json file.

```console
$ python3 similar_documents.py --similarity_func lexical --query "Which bank offers the best mortgage" --index_source json --index_path ./test_data/indices.json --n 3
```

### Load sentence embedding model from Huggingface and save to the repository

```console
$ python3 load_embedding_model.py
```