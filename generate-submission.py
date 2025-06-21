import csv
import pandas as pd
from pyserini.search.lucene import LuceneSearcher

# === Paths ===
index_dir = "indexes/tf-idf-clean-index"
queries_path = "data/test_queries.csv"
output_path = "results/clean-bm25-default.csv"

# === Load Queries ===
queries_df = pd.read_csv(queries_path)
queries = queries_df.to_dict("records")  # List of {'QueryId': ..., 'Query': ...}

# === Initialize Searcher ===
searcher = LuceneSearcher(index_dir)
searcher.set_bm25()  # Set BM25 as the default scoring model
# searcher.set_qld()
# searcher.set_rm3()
# searcher.set_rocchio()

# === Search and Write to Submission File ===
with open(output_path, mode="w", newline="") as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(["QueryId", "EntityId"])

    for item in queries:
        qid = str(item["QueryId"])
        query = item["Query"]

        hits = searcher.search(query, k=100)

        for hit in hits:
            doc_id = hit.docid  # This is your EntityId
            writer.writerow([qid, doc_id])

print(f"[INFO] Submission file written to: {output_path}")
