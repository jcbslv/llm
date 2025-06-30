from sentence_transformers import SentenceTransformer
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

with open('Justifications1Col.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]
lowerTriangle = []
for i in range(0, len(documents), 4):
    chunk = documents[i:i+4]
    embeddings = model.encode(chunk, normalize_embeddings=True)
    
    similarities = model.similarity(embeddings, embeddings)
    similarities = [[j.item() for j in row] for row in similarities]
    
    for m in range(len(similarities)):
        for n in range(m):
            lowerTriangle.append(round(similarities[m][n], 4))

# Histogram Same Scenario ####################################################################################################

plt.figure(figsize=(9, 7))
plt.hist(lowerTriangle, 15, density=True, edgecolor='black')
plt.title(f'Distribution of Cosine Similarities\nWithin Same Scenarios\n{modelName}',pad=10)
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
# plt.savefig(f'results/{modelName}/graphs/sameScenHist.png')
plt.show()
# plt.close()

##############################################################################################################################

