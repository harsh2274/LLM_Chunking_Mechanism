"""new_relevance_checking

This module implements a greedy relevance-based chunk selection algorithm
for HotpotQA-style data. It encodes query and document chunks with a
SentenceTransformer, computes relevance and similarity signals, and
selects chunks under a budget using a multi-factor greedy objective.

Main artifacts:
- `greedy_selection`: budgeted greedy selector using relevance, redundancy,
  coverage and entity overlap signals.
- Neo4j helpers to store chunks and similarity edges (optional runtime).

See DOCUMENTATION.md for algorithm details and usage instructions.
"""

import json
import numpy as np
from statistics import mean
from neo4j import GraphDatabase
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# model = SentenceTransformer("intfloat/e5-base-v2")


driver = GraphDatabase.driver(
    "neo4j://localhost:7687",  # or your instance URI
    auth=("neo4j", "123456789")
)

def cosine(list_a,list_b):
    """Compute cosine similarity between two vectors.

    Args:
        list_a: 1-D array-like embedding.
        list_b: 1-D array-like embedding.

    Returns:
        Float cosine similarity in [-1, 1].
    """
    return np.dot(list_a,list_b) / (np.linalg.norm(list_a) * np.linalg.norm(list_b))

def answer_covered(selected_chunks, answer):
    """Check whether the `answer` string appears in any selected chunk.

    Args:
        selected_chunks: list of chunk text strings.
        answer: answer string to match.

    Returns:
        True if `answer` (case-insensitive substring) appears in at least
        one chunk, otherwise False.
    """
    answer = answer.lower()
    return any(answer in chunk.lower() for chunk in selected_chunks)


def precision(selected_chunks, answer):
    """Compute precision of selected chunks with respect to the answer.

    Precision = (# chunks containing the answer) / (# selected chunks).

    Args:
        selected_chunks: list of chunk text strings.
        answer: answer string.

    Returns:
        Float precision in [0,1].
    """
    if not selected_chunks:
        return 0

    relevant = sum(answer.lower() in chunk.lower() for chunk in selected_chunks)
    return relevant / len(selected_chunks)

def store_chunks_in_neo4j(chunks):
    """Persist a list of chunks into Neo4j as `:Chunk` nodes.

    Each chunk is stored with properties `id` (index) and `text`.
    Requires `driver` to be configured for a running Neo4j instance.
    """
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            session.run(
                "MERGE (c:Chunk {id: $id, text: $text})",
                id=i,
                text=chunk
            )

def store_edges_in_neo4j(score_matrix, threshold=0.7):
    """Create `SIMILAR` relationships in Neo4j from a similarity matrix.

    Args:
        score_matrix: 2D array-like similarity scores between chunks.
        threshold: minimum score to create an edge.
    """
    with driver.session() as session:
        for i in range(len(score_matrix)):
            for j in range(len(score_matrix)):
                if i != j and score_matrix[i][j] > threshold:
                    session.run(
                        """
                        MATCH (a:Chunk {id: $i}), (b:Chunk {id: $j})
                        MERGE (a)-[:SIMILAR {score: $score}]->(b)
                        """,
                        i=i, j=j, score=float(score_matrix[i][j])
                    )

def get_neighbors(chunk_id, limit=5):
    """Return neighbor chunk ids connected by `:SIMILAR` edges.

    Args:
        chunk_id: integer id of the chunk node.
        limit: maximum number of neighbors to return.

    Returns:
        List of neighbor chunk ids (ints).
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Chunk {id: $id})-[:SIMILAR]->(n)
            RETURN n.id AS id
            LIMIT $limit
            """,
            id=chunk_id,
            limit=limit
        )
        return [record["id"] for record in result]

def extract_from_hotpot(sample):
    """Extract query, document chunks, and answer from a HotpotQA sample.

    The function concatenates sentences grouped per title into a single
    chunk per title.

    Returns:
        (query, chunks, answer)
    """
    query = sample["question"]
    answer = sample["answer"]
    chunks = []

    titles = sample["context"]["title"]
    sentences = sample["context"]["sentences"]

    for title in range(len(titles)):
        text = " ".join(sentences[title])   # combine sentences
        chunks.append(text)

    return query, chunks, answer

def entity_overlap(query, chunk):
    """Simple entity/word overlap score between query and chunk.

    This returns the fraction of unique query tokens that appear in the
    chunk. It is a lightweight proxy for entity overlap.
    """
    query_words = set(query.lower().split())
    chunk_words = set(chunk.lower().split())
    return len(query_words & chunk_words) / len(query_words)

def greedy_selection(chunk_embeddings, query_embedding, s, L, B=30, lambda_=2.0, gamma = 0.5, delta = 0.3):
    """Select chunk indices using a budgeted greedy objective.

    Objective factors:
    - relevance r (cosine with current query embedding)
    - redundancy (sum of pairwise similarities to chosen set S)
    - coverage (prefers chunks that add information w.r.t S)
    - entity overlap with original query (lightweight signal)

    Args:
        chunk_embeddings: array-like of chunk embeddings.
        query_embedding: embedding vector for the query.
        s: pairwise similarity matrix (n x n) between chunks.
        L: array-like of chunk lengths (token/word counts).
        B: budget (sum of lengths) to respect.
        lambda_: weight for redundancy penalty.
        gamma: weight for coverage reward.
        delta: weight for entity overlap reward.

    Returns:
        List of selected chunk indices (ints), in selection order.
    """

    S = []
    budget_used = 0
    remaining = set(range(len(chunk_embeddings)))
    current_query_embedding = query_embedding.copy()

    while True:
        r = np.array([cosine(current_query_embedding, emb) for emb in chunk_embeddings])
        best_score = -np.inf
        best_i = None

        for i in remaining:

            if r[i] < 0.3:
                continue

            redundancy = sum(s[i][j] for j in S)
            
            coverage = sum((1 - s[i][j]) * r[i] for j in S)

            entity_score = entity_overlap(query, chunks[i])

            gain = r[i]**2 - lambda_ * redundancy + gamma * coverage + delta * entity_score

            if r[i] < np.mean(r):
                gain *= 0.5

            score = gain / L[i]

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        if len(S) > 1 and best_score <= 0:
            break

        # budget check
        if budget_used + L[best_i] <= B:
            S.append(best_i)
            budget_used += L[best_i]
            remaining.remove(best_i)

            current_query_embedding = (current_query_embedding + chunk_embeddings[best_i]) / 2
            
        else:
            break

        if len(S) == 1:
            neighbors = get_neighbors(best_i)

            for n in neighbors:
                if n not in S:
                    remaining.add(n)
   
    return S

def top_k_selection(r, k=3):
    """Return top-k indices by score vector `r` (highest first)."""
    return list(np.argsort(r)[-k:][::-1])

greedy_hits = 0
topk_hits = 0

greedy_precision_total = 0
topk_precision_total = 0

total = 0


dataset = load_dataset("hotpot_qa", "fullwiki", split="train")

small_data = dataset.select(range(50))

total_greedy_len = 0
total_topk_len = 0

for sample in small_data:

    query, chunks, answer = extract_from_hotpot(sample)

    print("\n========================")
    print("Query:", query)
    print("Answer:", answer)

    # chunks = chunks[:50]

    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode([query])[0]
    answer_embedding = model.encode([answer])[0]

    L = [len(chunk.split()) for chunk in chunks]
   
    r_raw = np.array([cosine(query_embedding, emb) for emb in chunk_embeddings])

    r_norm = (r_raw - r_raw.min()) / (r_raw.max() - r_raw.min() + 1e-8)

    r = r_norm ** 2

    query_parts = query.split()

    r_parts = []

    for word in query_parts:
        word_emb = model.encode([word])[0]
        r_word = np.array([cosine(word_emb, emb) for emb in chunk_embeddings])
        r_parts.append(r_word)

    r_multi = np.array(r_parts).max(axis=0)

    # combine
    r = 0.6 * r_norm + 0.4 * r_multi

    # length normalization
    r = r / (np.array(L) ** 0.3)

    n = len(chunks)
    s = np.zeros((n, n))

    with driver.session() as session:
        for i in range(n):
            for j in range(i+1,n):
                sim = cosine(chunk_embeddings[i], chunk_embeddings[j])

                if sim > 0.6:
                    # print(f"Edge: {i} <-> {j} | score={sim}")
                    
                    session.run(
                        """
                        MATCH (a:Chunk {id: $i}), (b:Chunk {id: $j})
                        MERGE (a)-[:SIMILAR {score: $score}]->(b)
                        MERGE (b)-[:SIMILAR {score: $score}]->(a)
                        """,
                        i=i, j=j, score=float(sim)
                    )
                


    L = np.array([len(chunk.split()) for chunk in chunks])

    store_chunks_in_neo4j(chunks)
    store_edges_in_neo4j(s)

    query_embedding = model.encode([query])[0]
    selected_indices = greedy_selection(chunk_embeddings, query_embedding, s, L, B=150, lambda_=2.0, gamma = 0.5, delta = 0.3)

    print("\n[Greedy Selection]")
    for i in selected_indices:
        print("-", chunks[i])


    topk_indices = top_k_selection(r, k=3)

    greedy_chunks = [chunks[i] for i in selected_indices]
    topk_chunks = [chunks[i] for i in topk_indices]

    total += 1
    if answer_covered(greedy_chunks, answer):
        greedy_hits += 1

    if answer_covered(topk_chunks, answer):
        topk_hits += 1

    print("\n[Top-k Selection]")
    for i in topk_indices:
        print("-", chunks[i])

    greedy_precision_total += precision(greedy_chunks, answer)
    topk_precision_total += precision(topk_chunks, answer)

    total_greedy_len += len(greedy_chunks)
    total_topk_len += len(topk_chunks)

print("\n===== FINAL RESULTS =====")

print("Precision Greedy : ",greedy_precision_total/total)
print("Precision TopK : ",topk_precision_total/total)

print("Greedy Coverage:", greedy_hits / total)
print("Top-k Coverage:", topk_hits / total)

print("Average Greedy Chunks:", total_greedy_len / total)
print("Average Top-k Chunks:", total_topk_len / total)

# avg_greedy_len = total_greedy_chunks / total
# avg_topk_len = total_topk_chunks / total