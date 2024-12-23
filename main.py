from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

model = SentenceTransformer(
    # 'thenlper/gte-base')
    # 'thenlper/gte-large')
    # 'Mihaiii/gte-micro')
    # "Mihaiii/Ivysaur")
    # 'sentence-transformers/average_word_embeddings_glove.6B.300d')
    'sentence-transformers/average_word_embeddings_komninos')
    # 'sentence-transformers/all-mpnet-base-v2')
    # 'sentence-transformers/all-MiniLM-L12-v2')
    # 'sentence-transformers/sentence-t5-xl')
    # 'sentence-transformers/gtr-t5-xl')
    # "dunzhang/stella_en_1.5B_v5", trust_remote_code=True)
    # "dunzhang/stella_en_400M_v5",
    # trust_remote_code=True,
    # device="cpu",
    # config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False})

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
    print('\n',type(embeddings), embeddings.shape, i)
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
    plt.savefig(f'gte-base/plot{i}.png')  # Save plot to file



