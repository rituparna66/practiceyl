from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading Embedding Model...")
model = SentenceTransformer("all-MiniLM-l6-v2")
print("Model loaded\n")

print("-" * 55)
print("step 2 - single sentence embedding")
print("-" * 55)

sentence = " Gold price surged due to inflation fears"
embedding = model.encode(sentence)

print(f"sample : {embedding[:5].round(4)}")

print("-" * 55)
print(" cosine similarity heatmap")
print("-" * 55)

sentences = [
    "Gold price surged due to inflation fears",
    "XAU/USD rose sharply amid economic uncertainty",
    "The cat sat on the mat",
    "Federal reserve raised interest rates today",
    "Bitcoin hit all time high",
]
embeddings = model.encode(sentences)
print(f"embeddings shape:{embeddings.shape}")

sim_matrix = cosine_similarity(embeddings)

plt.figure(figsize=(9,6))

sns.heatmap(sim_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=[s[:25] + "..." for s in sentences],
            yticklabels=[s[:25] + "..." for s in sentences])
plt.title("Sentence Similarity Matrix")
plt.tight_layout()
plt.show()
print()

with open("Synthetic_Jyotisha_RAG_Corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()
