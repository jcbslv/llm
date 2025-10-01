from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np
import matplotlib.pyplot as plt
from chooseModel import chooseModel

modelName = chooseModel()   
model = SentenceTransformer(modelName)
    
##########################################################################################

# Load text
with open('Justifications1Col.txt', 'r') as file:
    documents = file.readlines()

documents = [line.strip() for line in documents]
print(documents)