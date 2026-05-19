# import json
# import numpy as np
# from statistics import mean
# from neo4j import GraphDatabase
# from datasets import load_dataset
# from scipy.stats import ttest_rel

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('all-MiniLM-L6-v2')
# # model = SentenceTransformer("intfloat/e5-base-v2")

# driver = GraphDatabase.driver(
#     "neo4j://localhost:7687",  # or your instance URI
#     auth=("neo4j", "123456789")
# )


# def compute_similarity_matrix(chunk_embeddings):

#     size_chunks = len(chunk_embeddings)

#     similarity_matrix = np.zeros((size_chunks, size_chunks))

#     for i in range(size_chunks):
#         for j in range(size_chunks):

#             similarity_matrix[i][j] = cosine(chunk_embeddings[i], chunk_embeddings[j])

#     return similarity_matrix


# def cosine(list_a,list_b):
#     return np.dot(list_a,list_b) / (np.linalg.norm(list_a) * np.linalg.norm(list_b))


# def answer_covered(selected_chunks, answer):
#     answer = answer.lower()
#     return any(answer in chunk.lower() for chunk in selected_chunks)


# def precision(selected_chunks, answer):
#     if not selected_chunks:
#         return 0
    
#     relevant = sum(answer.lower() in chunk.lower() for chunk in selected_chunks)
#     return relevant / len(selected_chunks)


# def recall(selected_chunks, all_chunks, answer):

#     relevant_total = sum(
#         answer.lower() in chunk.lower()
#         for chunk in all_chunks
#     )

#     if relevant_total == 0:
#         return 0

#     relevant_retrieved = sum(
#         answer.lower() in chunk.lower()
#         for chunk in selected_chunks
#     )

#     return relevant_retrieved / relevant_total


# def store_chunks_in_neo4j(chunks):
#     with driver.session() as session:
#         for i, chunk in enumerate(chunks):
#             session.run(
#                 "MERGE (c:Chunk {id: $id, text: $text})",
#                 id = i,
#                 text = chunk
#             )


# def store_edges_in_neo4j(score_matrix, threshold=0.7):
#     with driver.session() as session:
#         for i in range(len(score_matrix)):
#             for j in range(len(score_matrix)):
#                 if i != j and score_matrix[i][j] > threshold:
#                     session.run(
#                         """
#                         MATCH (a:Chunk {id: $i}), (b:Chunk {id: $j})
#                         MERGE (a)-[:SIMILAR {score: $score}]->(b)
#                         """,
#                         i = i, j = j, score=float(score_matrix[i][j])
#                     )


# def get_neighbors(chunk_id, limit=5):
#     with driver.session() as session:
#         result = session.run(
#             """
#             MATCH (c:Chunk {id: $id})-[:SIMILAR]->(n)
#             RETURN n.id AS id
#             LIMIT $limit
#             """,
#             id=chunk_id,
#             limit=limit
#         )
#         return [record["id"] for record in result]


# def extract_from_hotpot(sample):
   
#     query = sample["question"]
#     answer = sample["answer"]
   
#     chunks = []

#     titles = sample["context"]["title"]
#     sentences = sample["context"]["sentences"]

#     for title in range(len(titles)):
#         text = " ".join(sentences[title])
#         chunks.append(text)

#     return query, chunks, answer


# def extract_from_musique(sample):
#     query = sample["question"]
#     answer = sample["answer"]
    
#     chunks = []
#     for para in sample["paragraphs"]:
#         chunks.append(para["paragraph_text"])
    
#     return query, chunks, answer


# def extract_2wiki(sample):

#     query = sample["question"]
#     answer = sample["answer"]

#     chunks = []

#     context = sample["context"]

#     titles = context["title"]

#     contents = context["content"]
 
#     for title, sentences in zip(titles, contents):

#         text = title + " : "

#         if isinstance(sentences, list):

#             text += " ".join(
#                 str(s) for s in sentences
#             )

#         else:

#             text += str(sentences)

#         if len(text.strip()) > 0:

#             chunks.append(text)

#     return query, chunks, answer


# def entity_overlap(query, chunk):
#     query_words = set(query.lower().split())
#     chunk_words = set(chunk.lower().split())
#     return len(query_words & chunk_words) / (len(query_words) + 1e-8)


# def retrieval_f1(precision, recall):

#     if precision + recall == 0:
#         return 0

#     return 2 * precision * recall / (precision + recall)


# def mmr_selection(r, s, k=3, lambda_=0.5):
    
#     S = []
#     remaining = list(range(len(r)))

#     while len(S) < k and remaining:

#         best_score = -np.inf
#         best_i = None

#         for i in remaining:

#             if len(S) == 0:
#                 score = r[i]
#             else:
#                 max_sim = max(s[i][j] for j in S)
#                 score = lambda_ * r[i] - (1 - lambda_) * max_sim

#             if score > best_score:
#                 best_score = score
#                 best_i = i

#         S.append(best_i)
#         remaining.remove(best_i)

#     return S


# def colbert_style_rerank(
#     r,
#     chunks,
#     query,
#     k=3
# ):

#     scores = []

#     for i in range(len(chunks)):

#         entity = entity_overlap(
#             query,
#             chunks[i]
#         )

#         score = 0.7 * r[i] + 0.3 * entity

#         scores.append(score)

#     scores = np.array(scores)

#     return np.argsort(scores)[-k:][::-1]


# def greedy_selection(
#         chunks, 
#         chunk_embeddings, 
#         query_embedding, 
#         similarity_matrix, 
#         length_chunks, 
#         scoring_model, 
#         query,
#         B):
    
#     similarity_vector = []
#     budget_used = 0
#     remaining = set(range(len(chunk_embeddings)))

#     current_query_embedding = query_embedding.copy()

#     while True:

#         r_raw = np.array([
#             cosine(current_query_embedding, emb) 
#             for emb in chunk_embeddings
#         ])

#         r_norm = (r_raw - r_raw.min()) / (r_raw.max() - r_raw.min() + 1e-8)

#         relevance_vector_norm = r_norm ** 2

#         best_score = -np.inf
#         best_i = None

#         for i in remaining:

#             if relevance_vector_norm[i] < 0.3:
#                 continue

#             redundancy = 0

#             if len(similarity_vector) > 0:
#                 redundancy = np.mean([
#                     similarity_matrix[i][j] for j in similarity_vector
#                 ])

#             coverage = 0

#             if len(similarity_vector) > 0:
#                 coverage = np.sum([
#                     (1 - similarity_matrix[i][j]) for j in similarity_vector
#                 ])

#             entity_score = entity_overlap(query, chunks[i]) 

#             length = length_chunks[i]

#             features = np.array([[
#                 relevance_vector_norm[i],
#                 redundancy,
#                 coverage,
#                 entity_score,
#                 length
#             ]])

#             gain = scoring_model.predict_proba(features)[0][1]

#             score = gain / ((length + 1e-8) ** 0.5)

#             if score > best_score:
#                 best_score = score
#                 best_i = i

#         if best_i is None:
#             break

#         if best_score <= 0:
#             break

#         # budget check
#         if budget_used + length_chunks[best_i] <= B:
#             similarity_vector.append(best_i)
#             budget_used += length_chunks[best_i]
#             remaining.remove(best_i)

#             current_query_embedding = (current_query_embedding + chunk_embeddings[best_i]) / 2
            
#         else:
#             break

#         if len(similarity_vector) == 1:
#             neighbors = get_neighbors(best_i)

#             for n in neighbors:
#                 if n not in similarity_vector:
#                     remaining.add(n)
   
#     return similarity_vector


# def top_k_selection(r, k=3):
#     return list(np.argsort(r)[-k:][::-1])


# hotpot = load_dataset("hotpot_qa", "fullwiki", split="train")
# small_data_hotpot = hotpot.select(range(50))

# musique = load_dataset("dgslibisey/MuSiQue")
# small_data_musique = musique["train"].select(range(50))

# wiki = load_dataset("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35")
# small_data_wiki = wiki["train"].select(range(50))

# print(small_data_hotpot)

# X_train = []
# y_train = []


# for sample in small_data_hotpot:

#     query, chunks, answer = extract_from_hotpot(sample)


#     if len(chunks) == 0:
#         continue

#     chunk_embeddings = model.encode(chunks)
#     query_embedding = model.encode([query])[0]

#     relevance_vector = np.array([cosine(query_embedding, emb) for emb in chunk_embeddings])

#     size_chunks = len(chunks)

#     similarity_matrix = compute_similarity_matrix(chunk_embeddings)

#     with driver.session() as session:
#         for i in range(size_chunks):
#             for j in range(i+1,size_chunks):
#                 sim = cosine(chunk_embeddings[i], chunk_embeddings[j])

#                 if sim > 0.6:
                    
#                     session.run(
#                         """
#                         MATCH (a:Chunk {id: $i}), (b:Chunk {id: $j})
#                         MERGE (a)-[:SIMILAR {score: $score}]->(b)
#                         MERGE (b)-[:SIMILAR {score: $score}]->(a)
#                         """,
#                         i=i, j=j, score=float(sim)
#                     )
                
#     store_chunks_in_neo4j(chunks)

#     store_edges_in_neo4j(similarity_matrix)

#     titles = sample["context"]["title"]

#     support_titles = set(
#             sample["supporting_facts"]["title"]
#         )

#     for i in range(len(chunks)):

#         relevance = relevance_vector[i]

#         redundancy = np.mean(similarity_matrix[i])

#         coverage = np.sum(1 - similarity_matrix[i])

#         entity = entity_overlap(query, chunks[i])

#         length = len(chunks[i].split())

#         features = [
#             relevance,
#             redundancy,
#             coverage,
#             entity,
#             length
#         ]

#         X_train.append(features)

#         title = titles[i]

#         label = int(title in support_titles)

#         y_train.append(label)



# for sample in small_data_musique:

#     query, chunks, answer = extract_from_musique(sample)


#     if len(chunks) == 0:
#         continue

#     chunk_embeddings = model.encode(chunks)
#     query_embedding = model.encode([query])[0]

#     relevance_vector = np.array([cosine(query_embedding, emb) for emb in chunk_embeddings])

#     size_chunks = len(chunks)


#     similarity_matrix = compute_similarity_matrix(chunk_embeddings)

#     store_chunks_in_neo4j(chunks)

#     store_edges_in_neo4j(similarity_matrix)

#     with driver.session() as session:
#         for i in range(size_chunks):
#             for j in range(i+1,size_chunks):
#                 sim = cosine(chunk_embeddings[i], chunk_embeddings[j])

#                 if sim > 0.6:
                    
#                     session.run(
#                         """
#                         MATCH (a:Chunk {id: $i}), (b:Chunk {id: $j})
#                         MERGE (a)-[:SIMILAR {score: $score}]->(b)
#                         MERGE (b)-[:SIMILAR {score: $score}]->(a)
#                         """,
#                         i=i, j=j, score=float(sim)
#                     )

#     for i in range(len(chunks)):

#         relevance = relevance_vector[i]

#         redundancy = np.mean(similarity_matrix[i])

#         coverage = np.sum(1 - similarity_matrix[i])

#         entity = entity_overlap(query, chunks[i])

#         length = len(chunks[i].split())

#         features = [
#             relevance,
#             redundancy,
#             coverage,
#             entity,
#             length
#         ]

#         X_train.append(features)

#         label = int(answer.lower() in chunks[i].lower())

#         y_train.append(label)



# for sample in small_data_wiki:

#     query, chunks, answer = extract_2wiki(sample)


#     if len(chunks) == 0:
#         continue

#     chunk_embeddings = model.encode(chunks)
#     query_embedding = model.encode([query])[0]

#     relevance_vector = np.array([cosine(query_embedding, emb) for emb in chunk_embeddings])

#     size_chunks = len(chunks)

#     similarity_matrix = compute_similarity_matrix(chunk_embeddings)

#     with driver.session() as session:
#         for i in range(size_chunks):
#             for j in range(i+1,size_chunks):
#                 sim = cosine(chunk_embeddings[i], chunk_embeddings[j])

#                 if sim > 0.6:
                    
#                     session.run(
#                         """
#                         MATCH (a:Chunk {id: $i}), (b:Chunk {id: $j})
#                         MERGE (a)-[:SIMILAR {score: $score}]->(b)
#                         MERGE (b)-[:SIMILAR {score: $score}]->(a)
#                         """,
#                         i=i, j=j, score=float(sim)
#                     )
                
#     store_chunks_in_neo4j(chunks)

#     store_edges_in_neo4j(similarity_matrix)

#     for i in range(len(chunks)):

#         relevance = relevance_vector[i]

#         redundancy = np.mean(similarity_matrix[i])

#         coverage = np.sum(1 - similarity_matrix[i])

#         entity = entity_overlap(query, chunks[i])

#         length = len(chunks[i].split())

#         features = [
#             relevance,
#             redundancy,
#             coverage,
#             entity,
#             length
#         ]

#         X_train.append(features)

#         label = int(answer.lower() in chunks[i].lower())

#         y_train.append(label)


# scoring_model = LogisticRegression(max_iter=1000)

# scoring_model.fit(X_train, y_train)


# greedy_hits = 0
# topk_hits = 0
# mmr_hits = 0

# greedy_precision_total = 0
# topk_precision_total = 0
# mmr_precision_total = 0

# greedy_recall_total = 0
# topk_recall_total = 0
# mmr_recall_total = 0 

# total = 0


# total_greedy_len = 0
# total_topk_len = 0
# total_mmr_len = 0

# greedy_scores = []
# topk_scores = []
# mmr_scores = []


# # hotpot = load_dataset("hotpot_qa", "fullwiki", split="train")
# # data_hotpot = hotpot.select(range(50))

# # musique = load_dataset("dgslibisey/MuSiQue")
# # data_musique = musique["train"].select(range(50))

# wiki = load_dataset("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35")
# data_wiki = wiki["train"].select(range(50))



# for sample in data_wiki:

#     query, chunks, answer = extract_2wiki(sample)

#     if len(chunks) == 0:
#         continue

#     chunk_embeddings = model.encode(chunks)
#     query_embedding = model.encode([query])[0]
#     answer_embedding = model.encode([answer])[0]

#     relevance_vector = np.array([
#         cosine(query_embedding, emb) 
#         for emb in chunk_embeddings
#     ])

#     size_chunks = len(chunks)

#     similarity_matrix = compute_similarity_matrix(chunk_embeddings)

#     length_chunks = [
#         len(chunk.split())
#         for chunk in chunks
#     ]

#     query_embedding = model.encode([query])[0]

#     query_len = len(query.split())
#     num_chunks = len(chunks)


#     selected_indices = greedy_selection(chunks, chunk_embeddings, query_embedding, similarity_matrix, 
#                                         length_chunks, scoring_model, query, B=150)
    


#     print("\n[Greedy Selection]")
#     for i in selected_indices:
#         print("-", chunks[i])

#     topk_indices = top_k_selection(relevance_vector, k=3)
#     mmr_indices = mmr_selection(relevance_vector, similarity_matrix, k=3, lambda_=0.5)

#     greedy_chunks = [chunks[i] for i in selected_indices]
#     topk_chunks = [chunks[i] for i in topk_indices]
#     mmr_chunks = [chunks[i] for i in mmr_indices]

#     total += 1
    
#     if answer_covered(greedy_chunks, answer):
#         greedy_hits += 1

#     if answer_covered(topk_chunks, answer):
#         topk_hits += 1

#     if answer_covered(mmr_chunks, answer):
#         mmr_hits += 1

#     print("\n[Top-k Selection]")
    
#     for i in topk_indices:
#         print("-", chunks[i])

#     greedy_p = precision(greedy_chunks, answer)
#     topk_p = precision(topk_chunks, answer)
#     mmr_p = precision(mmr_chunks, answer)

#     greedy_scores.append(greedy_p)
#     topk_scores.append(topk_p)
#     mmr_scores.append(mmr_p)

#     greedy_precision_total += greedy_p
#     topk_precision_total += topk_p
#     mmr_precision_total += mmr_p

#     total_greedy_len += len(greedy_chunks)
#     total_topk_len += len(topk_chunks)
#     total_mmr_len += len(mmr_chunks)


#     greedy_recall_total += recall(greedy_chunks, chunks, answer)
#     topk_recall_total += recall(topk_chunks, chunks, answer)
#     mmr_recall_total += recall(mmr_chunks, chunks, answer)


# greedy_f1 = retrieval_f1(
#     greedy_precision_total / total,
#     greedy_recall_total / total
# )

# topk_f1 = retrieval_f1(
#     topk_precision_total / total,
#     topk_recall_total / total
# )

# mmr_f1 = retrieval_f1(
#     mmr_precision_total / total,
#     mmr_recall_total / total
# )


# print("\n===== FINAL RESULTS =====")

# print("Precision Greedy : ",greedy_precision_total/total)
# print("Precision TopK : ",topk_precision_total/total)
# print("Precision MMR : ",mmr_precision_total/total)

# print("Greedy Coverage:", greedy_hits / total)
# print("Top-k Coverage:", topk_hits / total)
# print("MMR Coverage:", mmr_hits / total)

# print("Recall Greedy : ", greedy_recall_total / total)
# print("Recall TopK : ", topk_recall_total / total)
# print("Recall MMR : ", mmr_recall_total / total)

# print("F1 Greedy :", greedy_f1)
# print("F1 TopK :", topk_f1)
# print("F1 MMR :", mmr_f1)

# print("Average Greedy Chunks:", total_greedy_len / total)
# print("Average Top-k Chunks:", total_topk_len / total)
# print("Average MMR Chunks:", total_mmr_len / total)



# print("\n===== STATISTICAL VALIDATION =====\n")

# # ---------------------------------------------------------
# # GREEDY vs TOP-K
# # ---------------------------------------------------------

# t_stat_topk, p_value_topk = ttest_rel(
#     greedy_scores,
#     topk_scores
# )

# print("Greedy vs Top-k")

# print("T-statistic :", t_stat_topk)
# print("P-value     :", p_value_topk)

# if p_value_topk < 0.05:
#     print("Result: Statistically Significant")
# else:
#     print("Result: NOT Statistically Significant")

# print()

# # ---------------------------------------------------------
# # GREEDY vs MMR
# # ---------------------------------------------------------

# t_stat_mmr, p_value_mmr = ttest_rel(
#     greedy_scores,
#     mmr_scores
# )

# print("Greedy vs MMR")

# print("T-statistic :", t_stat_mmr)
# print("P-value     :", p_value_mmr)

# if p_value_mmr < 0.05:
#     print("Result: Statistically Significant")
# else:
#     print("Result: NOT Statistically Significant")

# print()

# # =========================================================
# # OPTIONAL: MEAN + STD
# # =========================================================

# print("===== SCORE DISTRIBUTION =====")

# print(
#     "Greedy Mean:",
#     np.mean(greedy_scores),
#     "Std:",
#     np.std(greedy_scores)
# )

# print(
#     "Top-k Mean:",
#     np.mean(topk_scores),
#     "Std:",
#     np.std(topk_scores)
# )

# print(
#     "MMR Mean:",
#     np.mean(mmr_scores),
#     "Std:",
#     np.std(mmr_scores)
# )




import json
import numpy as np
from statistics import mean
from neo4j import GraphDatabase
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer("intfloat/e5-base-v2")

driver = GraphDatabase.driver(
    "neo4j://localhost:7687",  # or your instance URI
    auth=("neo4j", "123456789")
)

def compute_similarity_matrix(chunk_embeddings):

    size_chunks = len(chunk_embeddings)

    similarity_matrix = np.zeros((size_chunks, size_chunks))

    for i in range(size_chunks):
        for j in range(size_chunks):

            similarity_matrix[i][j] = cosine(chunk_embeddings[i], chunk_embeddings[j])

    return similarity_matrix


def cosine(list_a,list_b):
    return np.dot(list_a,list_b) / (np.linalg.norm(list_a) * np.linalg.norm(list_b))

def answer_covered(selected_chunks, answer):
    answer = answer.lower()
    return any(answer in chunk.lower() for chunk in selected_chunks)


def precision(selected_chunks, answer):
    if not selected_chunks:
        return 0
    
    relevant = sum(answer.lower() in chunk.lower() for chunk in selected_chunks)
    return relevant / len(selected_chunks)

def store_chunks_in_neo4j(chunks):
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            session.run(
                "MERGE (c:Chunk {id: $id, text: $text})",
                id = i,
                text = chunk
            )

def store_edges_in_neo4j(score_matrix, threshold=0.7):
    with driver.session() as session:
        for i in range(len(score_matrix)):
            for j in range(len(score_matrix)):
                if i != j and score_matrix[i][j] > threshold:
                    session.run(
                        """
                        MATCH (a:Chunk {id: $i}), (b:Chunk {id: $j})
                        MERGE (a)-[:SIMILAR {score: $score}]->(b)
                        """,
                        i = i, j = j, score=float(score_matrix[i][j])
                    )

def get_neighbors(chunk_id, limit=5):
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
   
    query = sample["question"]
    answer = sample["answer"]
   
    chunks = []

    titles = sample["context"]["title"]
    sentences = sample["context"]["sentences"]

    for title in range(len(titles)):
        text = " ".join(sentences[title])
        chunks.append(text)

    return query, chunks, answer


def extract_from_musique(sample):
    query = sample["question"]
    answer = sample["answer"]
    
    chunks = []
    for para in sample["paragraphs"]:
        chunks.append(para["paragraph_text"])
    
    return query, chunks, answer

def extract_2wiki(sample):

    query = sample["question"]
    answer = sample["answer"]

    chunks = []

    context = sample["context"]

    titles = context["title"]

    contents = context["content"]
 
    for title, sentences in zip(titles, contents):

        text = title + " : "

        if isinstance(sentences, list):

            text += " ".join(
                str(s) for s in sentences
            )

        else:

            text += str(sentences)

        if len(text.strip()) > 0:

            chunks.append(text)

    return query, chunks, answer


def entity_overlap(query, chunk):
    query_words = set(query.lower().split())
    chunk_words = set(chunk.lower().split())
    return len(query_words & chunk_words) / (len(query_words) + 1e-8)


def mmr_selection(r, s, k=3, lambda_=0.5):
    
    S = []
    remaining = list(range(len(r)))

    while len(S) < k and remaining:

        best_score = -np.inf
        best_i = None

        for i in remaining:

            if len(S) == 0:
                score = r[i]
            else:
                max_sim = max(s[i][j] for j in S)
                score = lambda_ * r[i] - (1 - lambda_) * max_sim

            if score > best_score:
                best_score = score
                best_i = i

        S.append(best_i)
        remaining.remove(best_i)

    return S



def greedy_selection(chunks, chunk_embeddings, query_embedding, s, L, scoring_model, B=150, lambda_=2.0, gamma = 0.5, delta = 0.3):
    
    S = []
    budget_used = 0
    remaining = set(range(len(chunk_embeddings)))

    current_query_embedding = query_embedding.copy()

    while True:

        r_raw = np.array([
            cosine(query_embedding, emb) 
            for emb in chunk_embeddings
        ])

        r_norm = (r_raw - r_raw.min()) / (r_raw.max() - r_raw.min() + 1e-8)

        r = r_norm ** 2

        best_score = -np.inf
        best_i = None

        for i in remaining:

            if r[i] < 0.3:
                continue

            redundancy = 0

            if len(S) > 0:
                redundancy = np.mean([
                    s[i][j] for j in S
                ])

            coverage = 0

            if len(S) > 0:
                coverage = np.sum([
                    (1 - s[i][j]) for j in S
                ])

            entity_score = entity_overlap(query, chunks[i]) / (len(chunks[i].split()) + 1e-5)

            length = L[i]

            features = np.array([[
                r[i],
                redundancy,
                coverage,
                entity,
                length
            ]])


            # gain = r[i] - lambda_ * redundancy + gamma * coverage + delta * entity_score

            gain = scoring_model.predict_proba(features)[0][1]


            score = gain / (length + 1e-8)

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        if best_score <= 0:
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
    return list(np.argsort(r)[-k:][::-1])



# dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
# small_data = dataset.select(range(50))

# musique = load_dataset("dgslibisey/MuSiQue")
# small_data = musique["train"].select(range(50))

wiki = load_dataset("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35")
small_data = wiki["train"].select(range(50))

X_train = []
y_train = []


for sample in small_data:

    query, chunks, answer = extract_2wiki(sample)

    if len(chunks) == 0:
        continue

    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode([query])[0]

    relevance_vector = np.array([cosine(query_embedding, emb) for emb in chunk_embeddings])

    size_chunks = len(chunks)

    similarity_matrix = compute_similarity_matrix(chunk_embeddings)

    with driver.session() as session:
        for i in range(size_chunks):
            for j in range(i+1,size_chunks):
                sim = cosine(chunk_embeddings[i], chunk_embeddings[j])

                if sim > 0.6:
                    
                    session.run(
                        """
                        MATCH (a:Chunk {id: $i}), (b:Chunk {id: $j})
                        MERGE (a)-[:SIMILAR {score: $score}]->(b)
                        MERGE (b)-[:SIMILAR {score: $score}]->(a)
                        """,
                        i=i, j=j, score=float(sim)
                    )
                
    store_chunks_in_neo4j(chunks)

    store_edges_in_neo4j(similarity_matrix)

    for i in range(len(chunks)):

        relevance = relevance_vector[i]

        redundancy = np.mean(similarity_matrix[i])

        coverage = np.sum(1 - similarity_matrix[i])

        entity = entity_overlap(query, chunks[i])

        length = len(chunks[i].split())

        features = [
            relevance,
            redundancy,
            coverage,
            entity,
            length
        ]

        X_train.append(features)

        label = int(answer.lower() in chunks[i].lower())

        y_train.append(label)


scoring_model = LogisticRegression(max_iter=1000)

scoring_model.fit(X_train, y_train)


# dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
# small_data = dataset.select(range(1000))

# musique = load_dataset("dgslibisey/MuSiQue")
# small_data = musique["train"].select(range(100))

wiki = load_dataset("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35")
small_data = wiki["train"].select(range(100))


greedy_hits = 0
topk_hits = 0
mmr_hits = 0

greedy_precision_total = 0
topk_precision_total = 0
mmr_precision_total = 0

total = 0


total_greedy_len = 0
total_topk_len = 0
total_mmr_len = 0


for sample in small_data:

    query, chunks, answer = extract_2wiki(sample)

    if len(chunks) == 0:
        continue

    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode([query])[0]
    answer_embedding = model.encode([answer])[0]

    relevance_vector = np.array([
        cosine(query_embedding, emb) 
        for emb in chunk_embeddings
    ])

    size_chunks = len(chunks)

    similarity_matrix = compute_similarity_matrix(chunk_embeddings)

    length_chunks = [
        len(chunk.split())
        for chunk in chunks
    ]


    query_embedding = model.encode([query])[0]

    query_len = len(query.split())
    num_chunks = len(chunks)

    lambda_ = 1.5 + 0.5 * (query_len / 10)
    gamma   = 0.4 + 0.1 * (num_chunks / 50)
    delta   = 0.3


    selected_indices = greedy_selection(chunks, chunk_embeddings, query_embedding, similarity_matrix, 
                                        length_chunks, scoring_model, B=150, lambda_ = lambda_, 
                                        gamma = gamma, delta = delta)


    print("\n[Greedy Selection]")
    for i in selected_indices:
        print("-", chunks[i])

    topk_indices = top_k_selection(relevance_vector, k=3)
    mmr_indices = mmr_selection(relevance_vector, similarity_matrix, k=3, lambda_=0.5)

    greedy_chunks = [chunks[i] for i in selected_indices]
    topk_chunks = [chunks[i] for i in topk_indices]
    mmr_chunks = [chunks[i] for i in mmr_indices]

    total += 1
    
    if answer_covered(greedy_chunks, answer):
        greedy_hits += 1

    if answer_covered(topk_chunks, answer):
        topk_hits += 1

    if answer_covered(mmr_chunks, answer):
        mmr_hits += 1

    print("\n[Top-k Selection]")
    
    for i in topk_indices:
        print("-", chunks[i])

    greedy_precision_total += precision(greedy_chunks, answer)
    topk_precision_total += precision(topk_chunks, answer)
    mmr_precision_total += precision(mmr_chunks, answer)

    total_greedy_len += len(greedy_chunks)
    total_topk_len += len(topk_chunks)
    total_mmr_len += len(mmr_chunks)


print("\n===== FINAL RESULTS =====")

print("Precision Greedy : ",greedy_precision_total/total)
print("Precision TopK : ",topk_precision_total/total)
print("Precision MMR : ",mmr_precision_total/total)

print("Greedy Coverage:", greedy_hits / total)
print("Top-k Coverage:", topk_hits / total)
print("MMR Coverage:", mmr_hits / total)

print("Average Greedy Chunks:", total_greedy_len / total)
print("Average Top-k Chunks:", total_topk_len / total)
print("Average MMR Chunks:", total_mmr_len / total)