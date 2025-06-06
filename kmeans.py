from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from chooseModel import chooseModel

modelName = chooseModel()   
model = SentenceTransformer(modelName)
    # "dunzhang/stella_en_1.5B_v5", trust_remote_code=True)
    # "dunzhang/stella_en_400M_v5",
    # trust_remote_code=True,
    # device="cpu",
    # config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False})

##########################################################################################


with open('Justifications1Col.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]
embeddings = model.encode(documents, normalize_embeddings=False)

# #for e5 models
# queries = []
# for each in documents:
#     queries.append(f"query: {each}")
# embeddings = model.encode(queries, normalize_embeddings=False)

##########################################################################################

# reduce embedding dimensions
tsne = TSNE(n_components=2)
reduced_embeddings = tsne.fit_transform(embeddings)
# print(reduced_embeddings)
# print(reduced_embeddings.shape)

###########################################################################################

# do kmeans
kmeans = KMeans(n_clusters=99, random_state=0)
# Fit the model to your data
kmeans.fit(reduced_embeddings)
# Get the cluster labels for each point
labels = kmeans.labels_

# print("Cluster labels:")
# print(labels)
# for i in range(len(reduced_embeddings)):
#     print(reduced_embeddings[i])
plt.figure(figsize=(8, 6))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=150, c='Chartreuse', label='Centroids')
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels)
plt.title(f"K-means Clustering\n{modelName}", pad=10)

for i in range(0, len(reduced_embeddings), 4):
    # group by scenario
    points = np.array([reduced_embeddings[i], reduced_embeddings[i+1], reduced_embeddings[i+2], reduced_embeddings[i+3]])
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(points)
    # Get the centroid coordinates of the single cluster
    centroid_x, centroid_y = kmeans.cluster_centers_[0]
    plt.scatter(centroid_x, centroid_y, marker='*', s=50, c='gold', label='Centroids')
    # plot connection from centroid to components
    for i in range(len(points)):
        plt.plot([points[i, 0], centroid_x], [points[i, 1], centroid_y], 'PaleGoldenRod', linewidth=1, ls=':')

    # print("Centroid:", (centroid_x, centroid_y))

for i, (x, y) in enumerate(reduced_embeddings):
    plt.annotate(i, (x, y), textcoords="offset points", xytext=(0,10), ha='center')  
plt.show()

#########################################################################################
plt.close()

