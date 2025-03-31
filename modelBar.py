import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

lsaAllScenario = [0.56, 0.69, 0.59, 0.62, 0.56, 0.74]
lsaTruth0 = [0.55, 0.66, 0.51, 0.59, 0.59, 0.77]
lsaTruth1 = [0.59, 0.73, 0.66, 0.64, 0.55, 0.76]
glove = [0.80, 0.83, 0.81, 0.81, 0.79, 0.86]
mpnet = [0.76, 0.75, 0.79, 0.72, 0.73,0.85]
miniLM = [0.75, 0.74, 0.76, 0.70, 0.72, 0.86]
gtr_t5 = [0.81, 0.82, 0.82, 0.79, 0.79, 0.89]
sentence_t5 = [0.90, 0.90, 0.90, 0.88, 0.88, 0.93]
snowflake = [0.76, 0.75, 0.78, 0.71, 0.71, 0.85]
e5 = [0.91, 0.91, 0.91, 0.89, 0.89, 0.93]

xLabels = ["0ShotCoT/\n0Shot", "1Shot/\n0Shot", "1Shot/\n0ShotCoT", "2shot/\n0Shot", "2Shot/\n0ShotCoT", "2Shot/\n1Shot"]
# plt.figure(figsize=(9, 7))
method_means = pd.DataFrame({
    'LSA': lsaAllScenario,
    'GloVe': glove,
    'all-mpnet-base-v2': mpnet,
    'all-MiniLM-L12-v2': miniLM,
    'sentence-t5-xl': sentence_t5,
    'gtr-t5-xl': gtr_t5,
    'snowflake-arctic-embed-l-v2.0': snowflake,
    'e5-large-v2': e5,
    # 'Truth 0': lsaTruth0,
    # 'Truth 1': lsaTruth1,
    }, index=xLabels)

# print(df)
method_means.plot.bar(rot=0, figsize=(9, 7), xlabel='Justification Pairs', ylabel='Median Similarity Score', title='Median Similarity Scores Across Same Scenario Pairs', legend=True)
plt.show()
plt.close()