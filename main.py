from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

with open('Justifications1Col.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]
# for line in documents:
#     print(line)

for i in range(0, len(documents), 4):
    chunk = documents[i:i+4]
    for doc in chunk:
        print(doc)
    embeddings = model.encode(chunk, normalize_embeddings=True)
    print('\n',type(embeddings), embeddings.shape)
    # for e in embeddings:
    #     print(e)
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    print('------------------------------------------------------------------------------------------------------------')
    
    # heatmaps
    plt.figure(figsize=(6, 5))
    plt.imshow(similarities, cmap='Blues', interpolation='none')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Cosine Similarity Heatmap')

    # vector labels
    vector_labels = ['Justification 1', 'Justification 2', 'Justification 3', 'Justification 4']
    plt.xticks(ticks=np.arange(len(chunk)), labels=vector_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(chunk)), labels=vector_labels)

    plt.tight_layout()
    plt.savefig('plots/plot.png')  # Save the plot to a file



