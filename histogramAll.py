from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np
import matplotlib.pyplot as plt
from chooseModel import chooseModel

modelName = chooseModel()   
model = SentenceTransformer(modelName)
    # "dunzhang/stella_en_1.5B_v5", trust_remote_code=True)
    # "dunzhang/stella_en_400M_v5",
    # trust_remote_code=True,
    # device="cpu",
    # config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False})

##########################################################################################


with open('Justifications1Col.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]
embeddings = model.encode(documents, normalize_embeddings=False)

# #for e5 models
# queries = []
# for each in documents:
#     queries.append(f"query: {each}")
# embeddings = model.encode(queries, normalize_embeddings=False)

#########################################################################################

# Get cosine similarity
similarities = model.similarity(embeddings, embeddings)


lowerTriangle = []
for i in range(len(documents)):
    for j in range(i):
        lowerTriangle.append(similarities[i][j].item())
plt.figure(figsize=(9, 7))
plt.hist(lowerTriangle, 50, density=True, edgecolor='black')
plt.title(f'Distribution of All Cosine Similarities\n{modelName}')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
# plt.savefig(f'results/{modelName}/graphs/allScenHist.png')

plt.show()

#########################################################################################
plt.close()