from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from chooseModel import chooseModel


def histogramPairs(documents, model):
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

    return lowerTriangleAll, lowerTriangleSim

with open('Justifications1Col.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]
# modelName = chooseModel()
modelName = 'sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(modelName)
arr1, arr2 = histogramPairs(documents, model)
fig, axes = plt.subplots(nrows=2, ncols=4)
axes[0][0].hist(arr1, range=(-.1,1), bins=50, density=True, edgecolor='black')
axes[0][0].set_title('all-mpnet-base-v2\nAll Justification Pairs', fontweight='bold', fontsize=9, wrap=True)
axes[0][0].set_ylabel('Frequency')

axes[1][0].hist(arr2, range=(-.1,1), bins=15, density=True, edgecolor='black')
axes[1][0].set_title('Matching Scenarios Only', fontweight='bold', fontsize=8, wrap=True)
axes[1][0].set_ylabel('Frequency')
axes[1][0].set_xlabel('Similarity Score')

modelName = 'sentence-transformers/average_word_embeddings_glove.6B.300d'
model = SentenceTransformer(modelName)
arr3, arr4 = histogramPairs(documents, model)
axes[0][1].hist(arr3, range=(-.1,1), bins=50, density=True, edgecolor='black')
axes[0][1].set_title('average_word_embeddings_glove.6B.300d\nAll Justification Pairs', fontweight='bold', fontsize=9, wrap=True)
# axes[0][1].set_ylabel('Frequency')

axes[1][1].hist(arr4, range=(-.1,1), bins=15, density=True, edgecolor='black')
axes[1][1].set_title('Matching Scenarios Only', fontweight='bold', fontsize=8, wrap=True)
# axes[1][1].set_ylabel('Frequency')
axes[1][1].set_xlabel('Similarity Score')

modelName = 'sentence-transformers/sentence-t5-xl'
model = SentenceTransformer(modelName)
arr5, arr6 = histogramPairs(documents, model)
axes[0][2].hist(arr5, range=(-.1,1), bins=50, density=True, edgecolor='black')
axes[0][2].set_title('sentence-t5-xl\nAll Justification Pairs', fontweight='bold', fontsize=9, wrap=True)
# axes[0][2].set_ylabel('Frequency')

axes[1][2].hist(arr6, range=(-.1,1), bins=15, density=True, edgecolor='black')
axes[1][2].set_title('Matching Scenarios Only', fontweight='bold', fontsize=8, wrap=True)
# axes[1][2].set_ylabel('Frequency')
axes[1][2].set_xlabel('Similarity Score')

modelName = 'intfloat/e5-large-v2'
model = SentenceTransformer(modelName)
arr7, arr8 = histogramPairs(documents, model)
axes[0][3].hist(arr7, range=(-.1,1), bins=50, density=True, edgecolor='black')
axes[0][3].set_title('e5-large-v2\nAll Justification Pairs', fontweight='bold', fontsize=9, wrap=True)
# axes[0][3].set_ylabel('Frequency')

axes[1][3].hist(arr8, range=(-.1,1), bins=15, density=True, edgecolor='black')
axes[1][3].set_title('Matching Scenarios Only', fontweight='bold', fontsize=8, wrap=True)
# axes[1][3].set_ylabel('Frequency')
axes[1][3].set_xlabel('Similarity Score')

# plt.savefig(f'results/{modelName}/graphs/dualHist.png')

plt.show()

#########################################################################################
plt.close()