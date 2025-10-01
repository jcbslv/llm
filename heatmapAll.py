from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np
import matplotlib.pyplot as plt
from chooseModel import chooseModel

modelName = chooseModel()   
model = SentenceTransformer(modelName)
    
##########################################################################################

# Load text
with open('Justifications1Col.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]
embeddings = model.encode(documents, normalize_embeddings=False)

#########################################################################################

# Get cosine similarity
similarities = model.similarity(embeddings, embeddings)

# heatmaps
plt.figure(figsize=(9,7))
plt.imshow(similarities, cmap='Blues', interpolation='none')
plt.colorbar(label='Cosine Similarity')
plt.title('Cosine Similarity Heatmap')
plt.show()
plt.close()