from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]

with open('Justifications1Col.txt', 'r') as file:
    # Read all lines into a list
    lines = file.readlines()


lines = [line.strip() for line in lines]
# for line in lines:
#     print(line)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(lines)
print(type(embeddings))