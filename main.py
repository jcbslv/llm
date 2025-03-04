from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

modelName = str(
    # 'thenlper/gte-base')
    # 'thenlper/gte-large')
    # 'Mihaiii/gte-micro')
    # "Mihaiii/Ivysaur")
    # "Mihaiii/Bulbasaur")
    # "Mihaiii/Venusaur")
    # 'intfloat/e5-small-v2')
    # 'intfloat/e5-base-v2')
    # 'intfloat/e5-large-v2')
    # 'Snowflake/snowflake-arctic-embed-l-v2.0')
    # 'sentence-transformers/average_word_embeddings_glove.6B.300d')
    # 'sentence-transformers/average_word_embeddings_komninos')
    'sentence-transformers/all-mpnet-base-v2')
    # 'sentence-transformers/all-MiniLM-L12-v2')
    # 'sentence-transformers/sentence-t5-xl')
    # 'sentence-transformers/gtr-t5-xl')
    # "dunzhang/stella_en_1.5B_v5")
    # "dunzhang/stella_en_400M_v5")
   
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
    similarities = [[i.item() for i in row] for row in similarities]
    
    # print(similarities)
    # lowerTriangle = []
    for i in range(len(similarities)):
        for j in range(i):
            lowerTriangle.append(round(similarities[i][j], 4))
print(lowerTriangle)
plt.hist(lowerTriangle, 11, density=True)
plt.savefig(f'results/{modelName}/simHistogram.png')
plt.close()

pair1 = []
pair2 = []
pair3 = []
pair4 = []
pair5 = []
pair6 = []

for j in range(6):
    for i in range(j, len(lowerTriangle), 4):
        pair1.append(lowerTriangle[i])
print(pair1)






# # Create a figure and axis object
# fig, ax = plt.subplots()

# # Plot the bar graph
# ax.bar(range(len(lowerTriangle)), lowerTriangle)

# # Add title and labels
# ax.set_title('Bar Graph Example')
# ax.set_xlabel('Index')
# ax.set_ylabel('Value')

# plt.savefig(f'results/{modelName}/bar.png')  # Save plot to file
# plt.close()



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


