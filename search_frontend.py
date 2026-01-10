from flask import Flask, request, jsonify
import math
import pickle
import re
from collections import Counter, defaultdict
from inverted_index_gcp import InvertedIndex
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# =========================
# CONFIG
# =========================
BUCKET = "samihdiab-123456-bucket"

BODY_INDEX_DIR = "body_index"
BODY_INDEX_NAME = "body"

TITLE_INDEX_DIR = "title_index"
TITLE_INDEX_NAME = "title"

TITLES_PATH = "docs_title.pkl"

N_DOCS = 6348910



# BM25-lite param
K1 = 1.6

# =========================
# Tokenizer
# =========================
try:
    nltk.download("stopwords", quiet=True)
except:
    pass

english_stopwords = frozenset(stopwords.words("english"))
corpus_stopwords = {"category", "references", "also", "links", "external", "see", "thumb"}
all_stopwords = english_stopwords.union(corpus_stopwords)

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
ps = PorterStemmer()

def tokenize(text):
    tokens = [t.group().lower() for t in RE_WORD.finditer(text.lower())]
    return [ps.stem(t) for t in tokens if t not in all_stopwords]

# =========================
# Load from GCS
# =========================
def load_pickle_from_gcs(bucket_name, path_in_bucket):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path_in_bucket)
    return pickle.loads(blob.download_as_bytes())

print("Loading indices...")
body_index  = InvertedIndex.read_index(BODY_INDEX_DIR, BODY_INDEX_NAME, bucket_name=BUCKET)
title_index = InvertedIndex.read_index(TITLE_INDEX_DIR, TITLE_INDEX_NAME, bucket_name=BUCKET)
TITLES = load_pickle_from_gcs(BUCKET, TITLES_PATH)
print("Resources loaded successfully.")

def load_pagerank_from_gcs(bucket_name, gz_csv_path):
    """
    Loads a pagerank file saved as .csv.gz with two columns: doc_id, pagerank
    Returns dict[int, float]
    """
    from google.cloud import storage
    import gzip, csv, io

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gz_csv_path)

    data = blob.download_as_bytes()                 # bytes of .gz
    f = gzip.GzipFile(fileobj=io.BytesIO(data))     # decompressed stream
    text = io.TextIOWrapper(f, encoding="utf-8")

    pr = {}
    reader = csv.reader(text)
    for row in reader:
        if not row:
            continue
        doc_id = int(row[0])
        score = float(row[1])
        pr[doc_id] = score
    return pr

print("Loading PageRank...")
PAGERANK = load_pagerank_from_gcs(
    BUCKET,
    "pr/part-00000-9cbf0794-66c3-4cb9-92d3-01c1758fb2ca-c000.csv.gz"
)
print("PageRank loaded:", len(PAGERANK))
MAX_PR = max(PAGERANK.values()) if PAGERANK else 1.0




# =========================
# BM25-lite for BODY (no doc lengths)
# =========================
def bm25_idf(df, N):
    return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

def bm25lite_search_body(query, topk=100):
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    q_tf = Counter(q_tokens)
    scores = defaultdict(float)
    matched_terms = defaultdict(int)

    for term in q_tf.keys():
        df = body_index.df.get(term, 0)
        if df == 0:
            continue

        idf = bm25_idf(df, N_DOCS)

        pl = body_index.read_a_posting_list(BODY_INDEX_DIR, term, BUCKET)
        for doc_id, tf in pl:
            doc_id = int(doc_id)

            # tf saturation (BM25-like, but no length norm)
            scores[doc_id] += idf * (tf * (K1 + 1)) / (tf + K1)
            matched_terms[doc_id] += 1

    if not scores:
        return []

    # coordination preference (helps "machine learning")
    n_terms = len(set(q_tf.keys()))
    if n_terms >= 2:
        filtered = {d: s for d, s in scores.items() if matched_terms[d] >= 2}
        if filtered:
            scores = filtered

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [[doc_id, TITLES.get(doc_id, str(doc_id))] for doc_id, _ in ranked]

def search_best(query, topk=100):
    body = bm25lite_search_body(query, topk=300)
    if not body:
        return []

    base = {doc_id: (300 - i) for i, (doc_id, _) in enumerate(body)}

    # NEW: title bonus
    tcounts = title_match_count(query)
    TITLE_WEIGHT = 30.0   # tune
    PR_WEIGHT = 5.0

    final = {}
    for doc_id, rscore in base.items():
        pr_norm = PAGERANK.get(doc_id, 0.0) / MAX_PR
        title_bonus = tcounts.get(doc_id, 0)
        final[doc_id] = rscore + TITLE_WEIGHT * title_bonus + PR_WEIGHT * pr_norm

    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [[doc_id, TITLES.get(doc_id, str(doc_id))] for doc_id, _ in ranked]




def title_match_count(query):
    tokens = set(tokenize(query))
    counts = defaultdict(int)
    for term in tokens:
        if term not in title_index.df:
            continue
        pl = title_index.read_a_posting_list(TITLE_INDEX_DIR, term, BUCKET)
        for doc_id, _ in pl:
            counts[int(doc_id)] += 1
    return counts  # doc_id -> number of matched terms





# =========================
# TITLE search (binary count)
# =========================
def search_in_title(query, topk=100):
    tokens = set(tokenize(query))
    if not tokens:
        return []

    matches = defaultdict(int)
    for term in tokens:
        if term not in title_index.df:
            continue
        pl = title_index.read_a_posting_list(TITLE_INDEX_DIR, term, BUCKET)
        for doc_id, _ in pl:
            matches[int(doc_id)] += 1

    ranked = sorted(matches.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [[doc_id, TITLES.get(doc_id, str(doc_id))] for doc_id, _ in ranked]

# =========================
# Flask Routes
# =========================
app = Flask(__name__)

@app.route("/search_body")
def search_body():
    query = request.args.get("query", "")
    topk = int(request.args.get("k", 100))
    return jsonify(bm25lite_search_body(query, topk=topk)) if query else jsonify([])

@app.route("/search_title")
def search_title():
    query = request.args.get("query", "")
    topk = int(request.args.get("k", 100))
    return jsonify(search_in_title(query, topk=topk)) if query else jsonify([])

@app.route("/search")
def search():
    query = request.args.get("query", "")
    topk = int(request.args.get("k", 100))
    return jsonify(search_best(query, topk=topk)) if query else jsonify([])



@app.route("/search_pagerank")
def search_pagerank():
    topk = int(request.args.get("k", 100))
    ranked = sorted(PAGERANK.items(), key=lambda x: x[1], reverse=True)[:topk]
    return jsonify([[doc_id, TITLES.get(doc_id, str(doc_id))] for doc_id, _ in ranked])



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
