from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from chooseModel import chooseModel

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

modelName = chooseModel()
model = SentenceTransformer(modelName)
    # "dunzhang/stella_en_1.5B_v5", trust_remote_code=True)
    # "dunzhang/stella_en_400M_v5",
    # trust_remote_code=True,
    # device="cpu",
    # config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False})

with open('truth0.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]
lowerTriangle = []
for i in range(0, len(documents), 4):
    chunk = documents[i:i+4]
    # print(chunk)
    embeddings = model.encode(chunk, normalize_embeddings=True)
    similarities = model.similarity(embeddings, embeddings)
    similarities = [[i.item() for i in row] for row in similarities]
    
    for i in range(len(similarities)):
        for j in range(i):
            lowerTriangle.append(round(similarities[i][j], 4))

# print(lowerTriangle)
allPairs = []
seperatedPairs = [[],[],[],[],[],[]]
pairMedians = []

for j in range(6):
    for i in range(j, len(lowerTriangle), 6):
        seperatedPairs[j].append(lowerTriangle[i])

# for j in range(6):
#     for i in range(j, len(lowerTriangle), 6):
#         allPairs.append(lowerTriangle[i])

# for i in range(0, 594, 99):
#     seperatedPairs.append([allPairs[j+i] for j in range(99)])
# print(seperatedPairs)
for pair in seperatedPairs:
    pairMedians.append(round(np.median(pair), 4))

for median in pairMedians:
    print(median)
    
xLabels = ["0ShotCoT/\n0Shot", "1Shot/\n0Shot", "1Shot/\n0ShotCoT", "2shot/\n0Shot", "2Shot/\n0ShotCoT", "2Shot/\n1Shot"]
plt.figure(figsize=(9, 7))
plt.bar(xLabels, pairMedians)

plt.title(f'Median Similarity Scores Across Same Scenario Pairs\nWith Ground Truth 0\n{modelName}')
plt.xlabel('Justification Pairs')
plt.ylabel('Median Similarity Score')
addlabels(xLabels, pairMedians)
plt.savefig(f'results/{modelName}/graphs/truth0Bar.png')
plt.show()
# plt.savefig(f'results/{modelName}/bar.png')  # Save plot to file
plt.close()



   

