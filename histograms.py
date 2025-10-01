from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from chooseModel import chooseModel

modelName = chooseModel()   

model = SentenceTransformer(modelName)
# model = SentenceTransformer(modelName, trust_remote_code=True)

with open('Justifications1Col.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]

######################################################################################
# Array for Histogram of Justificatino Pairs with Matching Scenarios
lowerTriangleSim = []
for i in range(0, len(documents), 4):
    chunk = documents[i:i+4]
    embeddings = model.encode(chunk, normalize_embeddings=True)

    #for e5 models
    # queries = []
    # for each in chunk:
    #     queries.append(f"query: {each}")
    # embeddings = model.encode(queries, normalize_embeddings=True) 
  
    similarities = model.similarity(embeddings, embeddings)
    similarities = [[j.item() for j in row] for row in similarities]
    
    for m in range(len(similarities)):
        for n in range(m):
            lowerTriangleSim.append(round(similarities[m][n], 4))
# print(lowerTriangle)
######################################################################################
# Array for histogram of All Justification Pairs 
embeddings = model.encode(documents, normalize_embeddings=True)

#for e5 models
# queriesAll = []
# for each in documents:
#     queriesAll.append(f"query: {each}")
# embeddings = model.encode(queriesAll, normalize_embeddings=True)

# Get cosine similarity
similarities = model.similarity(embeddings, embeddings)

lowerTriangleAll = []
for i in range(len(documents)):
    for j in range(i):
        lowerTriangleAll.append(similarities[i][j].item())
#####################################################################################
# Plot Histograms Subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7))
axes[0].hist(lowerTriangleAll, range=(-.1,1), bins=50, density=False, edgecolor='black')
axes[0].set_title(f'{modelName}\nAll Justification Pairs', fontweight='bold')
axes[0].set_ylabel('Frequency')

axes[1].hist(lowerTriangleSim, range=(-.1,1), bins=50, density=False, edgecolor='black')
axes[1].set_title('Justification Pairs with Matching Scenarios Only', fontweight='bold')
axes[1].set_ylabel('Frequency')
axes[1].set_xlabel('Similarity Score')

# plt.savefig(f'results/{modelName}/graphs/dualHist.png')

plt.show()

#########################################################################################
plt.close()