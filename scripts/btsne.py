import torch
import torch.nn as nn
import numpy as np
#from skdata.mnist.views import OfficialImageClassification
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.manifold import TSNE
#from tsne import bh_sne
from sklearn.decomposition import PCA

# load up data
#data = OfficialImageClassification(x_dtype="float32")
#x_data = data.all_images
#y_data = data.all_labels


wordDict = {}

keyf = open("WORD_FACTOR.key", 'r')
lines = keyf.readlines()
for line in lines:
    w = line.strip().split("->")
    wordDict[int(w[1])] = w[0]


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(wordDict[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)



print "just here"

with open("model.pt", 'rb') as f:           # Load the saved model because training is done and we want the one that performed best on the validation data
        model = torch.load(f)


X = []
for i in range(5002):
  x = model.encoders[0].weight.data[i, :]
  temp = []
  for el in x:
    temp.append(el)
  X.append(np.array(temp))

XX = np.array(X)
X = np.reshape(XX, (5002,128))

X_embedded = TSNE(n_components=2).fit_transform(X)
print X_embedded.shape




plot_embedding(X_embedded, "t-SNE embedding of the hinglish words")


exit(1)

#X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

print pca_result.singular_values_
exit(1)

print pca_result[:0]
print pca_result[:1]

vis_x = pca_result[:1]
vis_y = pca_result[:2]

plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

exit()
# For speed of computation, only run on a subset
#n = 20000
#x_data = x_data[:n]
#y_data = y_data[:n]

# perform t-SNE embedding
#vis_data = bh_sne(x_data)

# plot the result
#vis_x = vis_data[:, 0]
#vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
