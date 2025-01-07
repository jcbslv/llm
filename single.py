from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np
import matplotlib.pyplot as plt

modelName = str(
    # 'thenlper/gte-base')
    # 'thenlper/gte-large')
    # 'Mihaiii/gte-micro')
    # "Mihaiii/Ivysaur")
    # "Mihaiii/Bulbasaur")
    # "Mihaiii/Venusaur")
    'intfloat/e5-small-v2')
    # 'intfloat/e5-base-v2')
    # 'intfloat/e5-large-v2')
    # 'Snowflake/snowflake-arctic-embed-l-v2.0')
    # 'sentence-transformers/average_word_embeddings_glove.6B.300d')
    # 'sentence-transformers/average_word_embeddings_komninos')
    # 'sentence-transformers/all-mpnet-base-v2')
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
model.similarity_fn_name = SimilarityFunction.COSINE
with open('Justifications1Col.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]


embeddings = model.encode(documents, normalize_embeddings=False)
# print('\n',type(embeddings), embeddings.shape)
    # for e in embeddings:
    #     print(e)
for i in range(0, len(documents), 4):
    # with open(f'singleResults/{modelName}/embeddings/embeddings{i}.txt', 'w') as file:
    #     for line in embeddings:
    #         file.write(f'{line}\n')
    embMatrix = embeddings[i:(i+4)]
    print('\n',type(embMatrix), embMatrix.shape)
    # print(embMatrix)

    similarities = model.similarity(embMatrix, embMatrix)
    with open(f'singleResults/{modelName}/similarities/similarities{i}.txt', 'w') as file:
        file.write(str(similarities))

    # heatmaps
    plt.figure(figsize=(6, 5))
    plt.imshow(similarities, cmap='Blues', interpolation='none')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Cosine Similarity Heatmap')

    # vector labels
    vector_labels = ['Justification 1', 'Justification 2', 'Justification 3', 'Justification 4']
    plt.xticks(ticks=np.arange(len(embMatrix)), labels=vector_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(embMatrix)), labels=vector_labels)

    plt.tight_layout()
    plt.savefig(f'singleResults/{modelName}/plots/plot{i}.png')  # Save plot to file
    plt.close()


