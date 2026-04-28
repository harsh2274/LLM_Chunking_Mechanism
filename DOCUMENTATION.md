# Relevance-Guided Chunk Selection (Greedy)

Overview

This project implements a budgeted greedy chunk selection algorithm designed
for multi-hop QA (HotpotQA-style) and document retrieval. It combines
semantic relevance (embedding cosine), redundancy penalties, coverage
rewards, and a simple entity/word-overlap signal to pick informative
chunks under a token-length budget.

Where to find the code

- Main implementation: [new_relevance_checking.py](new_relevance_checking.py)

Key components

- Embeddings: SentenceTransformer (`all-MiniLM-L6-v2` by default).
- Similarity: cosine similarity used for relevance and for chunk-chunk
  similarity (`s` matrix).
- Greedy selector: `greedy_selection(...)` considers relevance, redundancy,
  coverage, and entity overlap and selects chunks until the length budget
  `B` is exhausted.
- Neo4j helpers: optional functions to store chunks and SIMILAR edges in
  Neo4j for neighborhood expansion (`get_neighbors`).

Algorithm (high level)

1. Encode chunks and the query with a sentence embedding model.
2. Compute per-chunk relevance scores `r` = cosine(query, chunk).
3. Optionally compute per-word query signals and combine them into `r_multi`.
4. Normalize and combine signals (relevance, multi-word signal, length
   normalization).
5. Build pairwise chunk similarity matrix `s` (cosine between chunk embeddings)
   and store high-similarity edges (used as redundancy/coverage signals).
6. Run `greedy_selection`:
   - Maintain a current query embedding (updated as average with chosen chunk).
   - For each candidate chunk compute: gain = r^2 - lambda * redundancy + gamma * coverage + delta * entity_overlap.
   - Select the chunk with highest gain/length until budget `B` is used.
   - After first selection, optionally expand search with neighbors from Neo4j.

Function reference (summary)

- `greedy_selection(chunk_embeddings, query_embedding, s, L, B=30, lambda_=2.0, gamma=0.5, delta=0.3)`
  - Returns selected indices in order.
  - Parameters control the trade-offs: `lambda_` (redundancy penalty),
    `gamma` (coverage reward), `delta` (entity overlap reward), `B` (budget).

- `top_k_selection(r, k=3)` — baseline top-k by relevance.

- `store_chunks_in_neo4j(chunks)`, `store_edges_in_neo4j(score_matrix, threshold=0.7)`, `get_neighbors(chunk_id, limit=5)`
  — utilities to persist and query chunk similarity graph in Neo4j.

Practical notes

- The code uses a global `driver` configured for `neo4j://localhost:7687`.
  Change credentials or disable Neo4j calls if not available.
- The greedy routine expects `chunks` to be available in the caller scope
  (the current script builds `chunks` from HotpotQA samples).
- Length normalization uses power `0.3` on token counts; budget `B` is in
  words (sum of chunk lengths).

Dependencies

- Python 3.8+
- sentence-transformers
- numpy
- datasets
- neo4j Python driver (only if using Neo4j helpers)

Quick usage

1. Install deps (example):

```bash
pip install sentence-transformers numpy datasets neo4j
```

2. Run the script from the repository root:

```bash
python new_relevance_checking.py
```

Tuning suggestions

- Increase `B` to allow longer multi-hop evidence sets.
- Adjust `lambda_` to penalize redundancy more strongly.
- Increase `gamma` to reward coverage from diverse chunks.
- Replace the embedding model with a stronger encoder (uncomment
  `intfloat/e5-base-v2` or other models) for better semantic signals.

Questions or next steps

- I can add inline examples, unit tests for the selection objective, or
  refactor `greedy_selection` to accept `chunks` explicitly (recommended).

---
Generated documentation for the greedy relevance selection algorithm.
