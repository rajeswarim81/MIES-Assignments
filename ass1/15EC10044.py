import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

df=pd.read_excel('dataset.xlsx')
Y=df[df.columns[0:1]]
X=df[df.columns[1:2]]
plt.figure(num="1. Original Dataset")
plt.scatter(X,Y, label='True Position')

error = []
K = range(1,6)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(X)
    error.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
plt.figure(num="2. Error Variation")
plt.plot(K, error, 'bx-')

plt.xlabel('Variations in k')
plt.ylabel('Error')
plt.title('The Elbow Method')
plt.show()