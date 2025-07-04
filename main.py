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
    # for doc in chunk:
    #     print(doc)
    embeddings = model.encode(chunk, normalize_embeddings=True)
    # print('\n',type(embeddings), embeddings.shape, i)
    # for e in embeddings:
    #     print(e)

    # with open(f'results/{modelName}/embeddings/embeddings{i}.txt', 'w') as file:
    #     for line in embeddings:
    #         file.write(f'{line}\n')

    similarities = model.similarity(embeddings, embeddings)
    similarities = [[j.item() for j in row] for row in similarities]
    
    # print(similarities)
    # lowerTriangle = []
    for m in range(len(similarities)):
        for n in range(m):
            lowerTriangle.append(round(similarities[m][n], 4))
# print(lowerTriangle)

# Histogram Same Scenario ####################################################################################################

plt.figure(figsize=(9, 7))
plt.hist(lowerTriangle, 15, density=True)
plt.title(f'Distribution of Cosine Similarities\nWithin Same Scenarios\n{modelName}',pad=10)
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
# plt.savefig(f'results/{modelName}/graphs/sameScenHist.png')
# plt.close()

##############################################################################################################################

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

for pair in seperatedPairs:
    pairMedians.append(np.median(pair))

for median in pairMedians:
    print(median)
        

xLabels = ["0ShotCoT/\n0Shot", "1Shot/\n0Shot", "1Shot/\n0ShotCoT", "2shot/\n0Shot", "2Shot/\n0ShotCoT", "2Shot/\n1Shot"]
plt.figure(figsize=(9, 7))
plt.bar(xLabels, pairMedians)
    
plt.title(f'Median Similarity Scores Across Same Scenario Pairs\n{modelName}', pad=10)
plt.xlabel('Justification Pairs')
plt.ylabel('Median Similarity Score')
for i in range(len(xLabels)):
    plt.text(i, pairMedians[i], pairMedians[i], ha = 'center')
# addlabels(xLabels, pairMedians)
# plt.savefig(f'results/{modelName}/graphs/medianBar.png')  # Save plot to file
plt.show()
plt.close()



    # with open(f'results/{modelName}/similarities/similarities{i}.txt', 'w') as file:
    #     file.write(str(similarities))

    # # heatmaps
    # plt.figure(figsize=(6, 5))
    # plt.imshow(similarities, cmap='Blues', interpolation='none')
    # plt.colorbar(label='Cosine Similarity')
    # plt.title('Cosine Similarity Heatmap')

    # # vector labels
    # vector_labels = ['Justification 1', 'Justification 2', 'Justification 3', 'Justification 4']
    # plt.xticks(ticks=np.arange(len(chunk)), labels=vector_labels, rotation=45)
    # plt.yticks(ticks=np.arange(len(chunk)), labels=vector_labels)

    # plt.tight_layout()
    # plt.savefig(f'results/{modelName}/plots/plot{i}.png')  # Save plot to file
    # plt.close()


