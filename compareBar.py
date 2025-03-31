from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
            lowerTriangle.append(round(similarities[m][n], 2))

seperatedPairs = [[],[],[],[],[],[]]
pairMedians = []

for j in range(6):
    for i in range(j, len(lowerTriangle), 6):
        seperatedPairs[j].append(lowerTriangle[i])

for pair in seperatedPairs:
    pairMedians.append(np.median(pair))

for median in pairMedians:
    print(median)

lsaAllScenario = [0.56, 0.69, 0.59, 0.62, 0.56, 0.74]
lsaTruth0 = [0.55, 0.66, 0.51, 0.59, 0.59, 0.77]
lsaTruth1 = [0.59, 0.73, 0.66, 0.64, 0.55, 0.76]

xLabels = ["0ShotCoT/\n0Shot", "1Shot/\n0Shot", "1Shot/\n0ShotCoT", "2shot/\n0Shot", "2Shot/\n0ShotCoT", "2Shot/\n1Shot"]
# plt.figure(figsize=(9, 7))

x = np.arange(len(xLabels))
width = 0.35

# fig, ax = plt.subplots()
plt.figure(figsize=(9, 7))
rect1 = plt.bar(x-width/2, lsaAllScenario, width, label='LSA')
plt.bar_label(rect1, padding=3)
rect2 = plt.bar(x+width/2, pairMedians, width, label='LLM')
plt.bar_label(rect2, padding=3)

plt.title(f'Median Similarity Scores Across Same Scenario Pairs\n{modelName}', pad=10)
plt.ylabel('Median Similarity Score')
plt.xlabel('Justification Pairs')
# ax.set_xticks(x + width, xLabels)
plt.xticks(x, xLabels)
plt.legend(loc='upper left', ncols=2)
plt.savefig(f'results/{modelName}/graphs/compareBar.png') 
plt.show()
plt.close()


