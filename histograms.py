from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np
import matplotlib.pyplot as plt

modelName = str(
    # 'thenlper/gte-base')
    # 'thenlper/gte-large')
    # 'Mihaiii/gte-micro')
    # 'Mihaiii/Ivysaur')
    # 'Mihaiii/Bulbasaur')
    # 'Mihaiii/Venusaur')
    # 'intfloat/e5-small-v2')
    # 'intfloat/e5-base-v2')
    # 'intfloat/e5-large-v2')
    # 'Snowflake/snowflake-arctic-embed-l-v2.0')
    'sentence-transformers/average_word_embeddings_glove.6B.300d')
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

similarities = model.similarity(embeddings, embeddings)

lowerTriangle = []
for i in range(len(documents)):
    for j in range(i):
        lowerTriangle.append(round(similarities[i][j].item(), 3))

plt.hist(lowerTriangle, 50)
plt.savefig(f'results/{modelName}/histogram.png')
plt.close()



