# Search engine project

## Supported usage

### Index a directory of pdf documents and store results into json

```console
$ python3 indexing.py --action write_indices --indexing_method bag_of_words --index_storing_format json --index_path ./test_data/indices.json --pdf_document_directory test_data
```