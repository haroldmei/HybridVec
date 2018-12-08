#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
PCA example with Iris Data-set
=========================================================

Principal Component Analysis applied to the Iris dataset.

See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information on this dataset.

"""
print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

from intrinsic_evals import get_embeddings
from web.embedding import Embedding

np.random.seed(5)

#centers = [[1, 1], [-1, -1], [1, -1]]
#iris = datasets.load_iris()
embeds,name = get_embeddings()

if isinstance(embeds, dict):
    embeds = Embedding.from_dict(embeds)

X = embeds.vectors
y = embeds.vocabulary.word_id

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

#for name in y:
#    ax.text3D(X[0], X[1] + 1.5, X[2], name,
#              horizontalalignment='center',
#              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results
#y = np.choose(y, [1, 2, 0]).astype(np.float)
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
#           edgecolor='k')
           
ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.nipy_spectral, edgecolor='k')

for i, txt in enumerate(y):
    ax.text(X[i,0],X[i,1],X[i,2], txt, size=10, zorder=1, color='k') 

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()
