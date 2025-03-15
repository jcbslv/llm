import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
with open('Justifications1Col.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]
# embeddings = model.encode(documents, normalize_embeddings=False)

#for e5 models
queries = []
for each in documents:
    queries.append(f"query: {each}")
# embeddings = model.encode(queries, normalize_embeddings=False)

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
model = AutoModel.from_pretrained('intfloat/e5-base-v2')

# Tokenize the input texts
batch_dict = tokenizer(queries, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
print(embeddings)

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
emb_detached = embeddings.detach()
similarities = cosine_similarity(emb_detached, emb_detached)

lowerTriangle = []
for i in range(len(documents)):
    for j in range(i):
        lowerTriangle.append(similarities[i][j].item())

plt.hist(lowerTriangle, 50, density=True)
# plt.savefig(f'results/{modelName}/histogram.png')

plt.show()
plt.close()
