import os
import json
import subprocess
from tqdm import tqdm

# === Paths ===
original_path = "data/corpus.jsonl"  # Input corpus file
input_dir = "data/json_corpus"  # Directory for preprocessed JSONL file
processed_path = os.path.join(input_dir, "docs.jsonl")  # Output JSONL file
# index_dir = "indexes/bm25-clean-index"  # Lucene index output directory
index_dir = "indexes/tf-idf-clean-index"  # Lucene index output directory

if not os.path.exists(processed_path):
    # === Step 1: Preprocess the corpus ===
    print("[INFO] Preprocessing corpus...")
    os.makedirs(input_dir, exist_ok=True)

    total_lines = 4641784

    def clean_text(text):
        """Basic text cleaning function."""
        # Remove extra whitespace
        text = " ".join(text.split())
        # Convert to lowercase
        text = text.lower()
        # Remove special characters (keep alphanumeric and spaces)
        text = "".join(char for char in text if char.isalnum() or char.isspace())
        return text

    # Process and convert each document
    with open(original_path, "r") as f_in, open(processed_path, "w") as f_out:
        for line in tqdm(f_in, total=total_lines, desc="[INFO] Processing documents"):
            doc = json.loads(line)
            contents = clean_text(f"{doc.get('title', '')}. {doc.get('text', '')}")
            if "keywords" in doc:
                contents += " " + " ".join(map(clean_text, doc["keywords"]))
            f_out.write(json.dumps({"id": doc["id"], "contents": contents}) + "\n")

    print(f"[INFO] Preprocessed corpus saved to: {processed_path}")

# === Step 2: Build the BM25 index using Pyserini (Lucene backend) ===
print("[INFO] Building BM25 index...")

cmd = [
    "python",
    "-m",
    "pyserini.index.lucene",  # ← correct module
    "--collection",
    "JsonCollection",
    "--input",
    input_dir,  # ← this must be a directory
    "--index",
    index_dir,
    "--generator",
    "DefaultLuceneDocumentGenerator",
    "--threads",
    "8",
    "--storePositions",
    "--storeDocvectors",
    "--storeRaw",
    "--verbose",
]

# Stream indexing output in real-time
process = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

for line in process.stdout:
    print(line, end="")

process.wait()
if process.returncode == 0:
    print(f"[INFO] Index successfully built at: {index_dir}")
else:
    print("[ERROR] Indexing failed.")
