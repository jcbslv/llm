from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load the dataset
file_path = "CSCoTGPT4.csv"  # Replace with your dataset's file path
df = pd.read_csv(file_path)

# List of justification columns to compare
justification_columns = [
    "BASE 0-SHOT JUSTIFICATION",
    "BASE 0-SHOT CoT JUSTIFICATION",
    "BASE 1-SHOT JUSTIFICATION",
    "BASE 2-SHOT JUSTIFICATION",
    "DETAILED 0-SHOT JUSTIFICATION",
    "DETAILED GPT4 0-SHOT CoT JUSTIFICATION",
    "DETAILED 1-SHOT JUSTIFICATIONS",
    "DETAILED 2-SHOT JUSTIFICATIONS",
    "DETAILED 1-SHOT JUSTIFICATION",
    "DETAILED 2-SHOT CoT JUSTIFICATION"
]

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# List to store all similarity matrices
all_similarity_matrices = []

# Process justifications for each prompt
for index, row in df.iterrows():
    # Extract justifications for the current prompt
    # print(row)
    justifications = row[justification_columns].fillna("").tolist()
    print(justifications)

    # Encode the justifications
    embeddings = model.encode(justifications)
    print(embeddings.shape)
    # Compute cosine similarity for all pairs
    similarities = model.similarity(embeddings, embeddings)

    # Ensure the matrix is 10x10
    # if similarities.shape != (10, 10):
    #     raise ValueError(f"Matrix is not 10x10 for prompt {index + 1}!")

    print(similarities)

    # Create a DataFrame for the current prompt's similarity matrix
    # similarity_matrix_df = pd.DataFrame(
    #     similarities,
    #     index=justification_columns,
    #     columns=justification_columns)

    # Print the similarity matrix for the current prompt
    # print(f"\nSimilarity Matrix for Prompt {index + 1} (Input: {row['input']}):")
    # print(similarity_matrix_df)

    # Append the similarity matrix to the list
    # all_similarity_matrices.append(similarities)

# Calculate the final median similarity matrix
# final_median_matrix = np.median(np.array(all_similarity_matrices), axis=0)

# Create a DataFrame for the final median similarity matrix
# final_median_df = pd.DataFrame(
#     final_median_matrix,
#     index=justification_columns,
#     columns=justification_columns)

# Show the final median similarity matrix
# print("\nFinal Median Similarity Matrix:")
# print(final_median_df)

# Save the final median similarity matrix to a CSV
# final_median_df.to_csv("final_justification_median_similarity_matrix.csv")

# print("\nFinal median similarity matrix saved!")