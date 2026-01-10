# 🔍 Wikipedia Search Engine at Scale

A **Wikipedia-scale information retrieval system** built from scratch as part of an academic search engine project.  
The system indexes millions of Wikipedia articles and serves ranked search results using a **hybrid relevance model** that combines **BM25 text ranking** with **PageRank authority scoring**.

> ⚡ Designed for accuracy, efficiency, and large-scale deployment on Google Cloud.

---

## 🧠 Motivation

Search engines are not just about matching words — they must balance:

- **Textual relevance** (Is this document about the query?)
- **Authority & importance** (Is this document trusted and influential?)
- **Efficiency** (Can we answer quickly at scale?)

This project explores these challenges by implementing a **real search engine pipeline**, end-to-end, on a large real-world corpus: **Wikipedia**.

---

## 🏗️ System Architecture

Query
↓
Tokenization & Stemming
↓
Inverted Index Lookup (GCS)
↓
BM25-lite Scoring (Body)
↓
PageRank Normalization
↓
Score Fusion
↓
Top-K Ranked Results


### Key Design Decisions
- Separate **body** and **title** indexes
- Store all indexes on **Google Cloud Storage**
- Avoid query-time calls to external services
- No caching — every query is processed live

---

## 🔬 Retrieval Models

### 📝 BM25-lite (Body Index)
- Term frequency saturation
- No document length normalization (for efficiency)
- Coordination bonus for multi-term queries

### 🏷️ Title Matching
- Binary term matching
- Improves precision for navigational queries

### 📈 PageRank Integration
- Precomputed PageRank scores
- Normalized and lightly weighted
- Improves authority without overpowering relevance


---

## 🚀 Features

- 🔎 Full-text search over millions of documents
- 🧮 Hybrid ranking: BM25 + PageRank
- ⚡ Efficient query processing (< 35s per query)
- 🌐 REST API using Flask
- ☁️ Cloud-native design (GCS + GCE)
- 📊 Quantitative & qualitative evaluation

---

## 🌐 API Endpoints

| Endpoint | Description |
|-------|-------------|
| `/search?query=...&k=10` | Best ranking (BM25 + PageRank) |
| `/search_body` | Body-only BM25 search |
| `/search_title` | Title-only matching |
| `/search_pagerank` | Pure PageRank ranking |

### Example

Queries can be issued via browser or `curl`.

---

## ☁️ Data & Storage

- **Indexes:** Google Cloud Storage
- **Corpus:** Wikipedia
- **Index types:**
  - Body inverted index
  - Title inverted index
  - PageRank table
  - Document titles mapping

All data is publicly accessible as required.

---

## 📊 Evaluation Methodology

### Metrics Used
- Precision@10
- Average Precision@10 (AP@10)
- Mean Average Precision@10 (MAP@10)

### Results Summary

| Metric | Value |
|-----|------|
| Mean Precision@10 | **0.71** |
| MAP@10 | **0.64** |
| Queries with AP@10 < 0.1 | **0 / 30** |

✔ All project quality requirements are satisfied.

---

## 📉 Error Analysis & Insights

- Lower-performing queries tend to be:
  - Broad or ambiguous
  - Historical or conceptual
- Causes:
  - Lack of semantic understanding
  - No phrase-level ranking
- Potential improvements:
  - Query expansion
  - Phrase indexing
  - Embedding-based reranking

---

## 🛠️ Technologies Used

- Python
- Flask
- Google Cloud Storage
- Google Compute Engine
- NLTK
- BM25
- PageRank

---

## 👨‍🎓 Author

- **Name:** YOUR NAME  
- **Student ID:** XXXXXXXX  
- **Email:** your_email@post.bgu.ac.il  

---

## 📎 Notes

This project was implemented without:
- External APIs at query time
- Caching of results
- External search services

All retrieval and ranking logic is executed locally.
