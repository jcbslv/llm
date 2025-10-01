from google import genai
import numpy as np
# from google.genai import types
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

client = genai.Client()
modelName = "gemini-embedding-001"
with open('justifications.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]

result = [
    np.array(e.values) for e in client.models.embed_content(
        model="gemini-embedding-001",
        contents=documents, 
        config=genai.types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
]

# Calculate cosine similarity. Higher scores = greater semantic similarity.

embeddings_matrix = np.array(result)
similarity_matrix = cosine_similarity(embeddings_matrix)

# for i, text1 in enumerate(documents):
#     for j in range(i + 1, len(documents)):
#         text2 = documents[j]
#         similarity = similarity_matrix[i, j]
#         print(f"Similarity between '{text1}' and '{text2}': {similarity:.4f}")

# similarities = cosine_similarity(result)
# print("Similarity matrix:")
# print(similarities)

lowerTriangle = []
for i in range(len(documents)):
    for j in range(i):
        lowerTriangle.append(similarity_matrix[i][j].item())

plt.figure(figsize=(9, 7))
plt.hist(lowerTriangle, 50, density=True, edgecolor='black')
plt.title(f'Distribution of All Cosine Similarities\n{modelName}')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
# plt.savefig(f'results/{modelName}/graphs/allScenHist.png')
plt.show()
plt.close()