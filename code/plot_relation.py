import numpy as np
import matplotlib.pyplot as plt

files = np.load("data/train_raw_samples.npz")
X = files['X']
y = files['y']
plt.scatter(X, y, alpha=0.5)
plt.savefig('plots/raw.png', bbox_inches='tight')
