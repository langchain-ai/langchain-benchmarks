# LangChain Docs Task 

This code contains utilities to scrape the LangChain docs (already run) and index them
using common techniques. The docs were scraped using the code in `_ingest_docs.py` and
uploaded to gcs. To better compare retrieval techniques, we hold these constant and pull
from that cache whenever generating different indices.


The content in `indexing` composes some common indexing strategies with default paramaters for
benchmarking on the langchain docs.