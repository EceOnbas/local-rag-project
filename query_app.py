import json
import math
import sqlite3
import re
from foundry_local_sdk import Configuration, FoundryLocalManager


DB_NAME = "rag_database.db"


# --- HELPER FUNCTIONS FOR HYBRID SEARCH & BM25 ---


def tokenize(text):
    """Metni kelimelerine ayırır."""
    return re.findall(r'\w+', text.lower())


def compute_bm25_score(query_tokens, doc_tokens, avg_doc_len, N, df, k1=1.5, b=0.75):
    """BM25 Skorlaması."""
    score = 0.0
    doc_len = len(doc_tokens)
    doc_freqs = {}
    for token in doc_tokens:
        doc_freqs[token] = doc_freqs.get(token, 0) + 1


    for q in query_tokens:
        if q not in doc_freqs:
            continue
        f = doc_freqs[q]
        n_q = df.get(q, 0)
        idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += idf * (numerator / denominator)
    return score


def cosine_similarity(a, b):
    """Vektör Benzerliği."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def hybrid_search(query, query_embedding, top_k=6):
    """BM25 + Vector Search birleşimi ve RRF Re-ranking."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()


    if not rows:
        return []


    N = len(rows)
    query_tokens = tokenize(query)
    doc_tokens_list = [tokenize(r[1]) for r in rows]
    avg_doc_len = sum(len(d) for d in doc_tokens_list) / N


    df = {}
    for d_tokens in doc_tokens_list:
        for term in set(d_tokens):
            df[term] = df.get(term, 0) + 1


    bm25_scores = []
    vector_scores = []


    for idx, (file_name, content, emb_str) in enumerate(rows):
        b_score = compute_bm25_score(query_tokens, doc_tokens_list[idx], avg_doc_len, N, df)
        bm25_scores.append((idx, b_score))


        doc_emb = json.loads(emb_str)
        v_score = cosine_similarity(query_embedding, doc_emb)
        vector_scores.append((idx, v_score))


    bm25_scores.sort(key=lambda x: x[1], reverse=True)
    vector_scores.sort(key=lambda x: x[1], reverse=True)


    rrf_scores = {}
    k = 60
    for rank, (idx, _) in enumerate(bm25_scores):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
    for rank, (idx, _) in enumerate(vector_scores):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))


    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


    final_results = []
    for idx, score in sorted_rrf[:top_k]:
        file_name, content, _ = rows[idx]
        final_results.append((file_name, content, score))


    return final_results


# --- MAIN APPLICATION ---


def main():
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance


    print("Loading embedding model...")
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()


    print("Searching for Phi-3.5 model in catalog...")
   
    chat_model = None
    for model in manager.catalog.list_models():
        if "phi-3.5" in model.id.lower():
            print(f"Found matching model: '{model.id}'")
            chat_model = model
            break


    if chat_model is None:
        raise RuntimeError("Phi-3.5 modeli kataloğunuzda bulunamadı.")


    # 1. Önce modeli bilgisayara indiriyoruz
    print("Downloading Phi-3.5-mini model (First time only, please wait)...")
    chat_model.download()


    # 2. İndirme bittikten sonra hafızaya yüklüyoruz
    print("Loading Phi-3.5-mini model into memory...")
    chat_model.load()
    chat_client = chat_model.get_chat_client()


    print("\n--- Optimized RAG System Ready (Phi-3.5-mini | top_k=6) ---")


    while True:
        query = input("Question: ").strip()
        if not query or query.lower() == "quit":
            break


        query_response = embedding_client.generate_embedding(query)
        query_emb = query_response.data[0].embedding


        top_matches = hybrid_search(query, query_emb, top_k=6)


        if not top_matches:
            print("\nAnswer: No documents found.\n")
            continue


        print("\n--- Retrieved Relevant Chunks ---")
        for idx, (file_name, content, score) in enumerate(top_matches, start=1):
            print(f"[{idx}] Source: {file_name} | RRF Score: {score:.4f}")
            print(f"    Preview: {content[:150].strip()}...\n")


        context_str = "\n\n".join([f"Passage {idx}:\n{doc[1]}" for idx, doc in enumerate(top_matches, start=1)])


        messages = [
    {
        "role": "system",
        "content": (
            "You are a strict, facts-only academic assistant. Answer using ONLY explicit facts from the context.\n\n"
            "RULES:\n"
            "1. Read table rows VERY carefully. Copy the EXACT Start Time and End Time without modifying them.\n"
            "2. Check if the course has MULTIPLE lecture days/sessions listed (e.g., Tue AND Fri). List ALL of them.\n"
            "3. Format every schedule line clearly: [Day]: [Start-End Time] | Classroom: [Place].\n"
            "4. Do NOT invent, assume, or modify any time intervals or classroom codes.\n"
            "5. Append a citation at the end of every sentence: [Source: SourceFile | Page: PageNumber].\n\n"
            f"Context Passages:\n{context_str}"
        )
    },
    {"role": "user", "content": query}
]


        print("Answer: ", end="", flush=True)
        for chunk in chat_client.complete_streaming_chat(messages):
            if chunk.choices and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
        print("\n" + "="*50 + "\n")


    embedding_model.unload()
    chat_model.unload()


if __name__ == "__main__":
    main()
