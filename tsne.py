import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

X_resampled = pickle.load(open( "X_resampled.pkl", "rb" ))
y_resampled = pickle.load(open( "y_resampled.pkl", "rb" ))

from sklearn.manifold import TSNE

model = TSNE(n_components=2, verbose=0, n_iter=300)
t = model.fit_transform(X_resampled)

pickle.dump(t, open( "tsne_model_resampled.pkl", "wb" ))
