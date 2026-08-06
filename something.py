import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

print("Loading Embedding Model...")
model = SentenceTransformer("all-MiniLM-l6-v2")
print("Model loaded\n")

with open("Synthetic_Jyotisha_RAG_Corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()


def chunk_text(text, model, chunk_size=500, overlap=50):
    tokenizer = model.tokenizer
    tokens = tokenizer.encode(text)
    token_len = len(tokens)

    chunks = []
    start = 0

    while start < token_len:
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


chunks = chunk_text(text, model, chunk_size=250, overlap=50)

chunk_embeddings = model.encode(chunks)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings).astype("float32"))

print(f"Indexed{index.ntotal} chunks")


def query_index(query, model, index, chunks, k=5):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "chunk": chunks[idx],
            "distance": float(distances[0][rank])
        })
    return results


# example usage
query = "What does Saturn Mahadasha indicate for career timing?"
results = query_index(query, model, index, chunks, k=5)

for r in results:
    print(f"[{r['rank']}] (dist={r['distance']:.4f}) {r['chunk'][:150]}...\n")

faiss.write_index(index, "jyotisha.index")

import pickle

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

# later, to reload without re-embedding:
index = faiss.read_index("jyotisha.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def build_prompt(query, results):
    context = "\n\n".join([f"[{r['rank']}] {r['chunk']}" for r in results])
    return f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""


results = query_index(query, model, index, chunks, k=5)
prompt = build_prompt(query, results)
# send `prompt` to your LLM (Claude API, GPT, etc.) and print the response


client = OpenAI()  # picks up ANTHROPIC_API_KEY from env


def generate_answer(query, results, model_name="gpt-4o-mini"):
    context = "\n\n".join([f"[{r['rank']}] {r['chunk']}" for r in results])

    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {query}"""

    response = client.chat.completions.create(
        model=model_name,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# full pipeline, end to end
query = "What does Saturn Mahadasha indicate for career timing?"
results = query_index(query, model, index, chunks, k=5)
answer = generate_answer(query, results)
print(answer)

