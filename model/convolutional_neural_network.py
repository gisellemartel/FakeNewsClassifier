import sys
sys.path.insert(1, '../')

import preprocess as preprocess
import tools as tools

#Library imports for deep-learning recommender
import numpy as np
import sklearn
import sklearn.preprocessing     # For StandardScaler
import sklearn.linear_model      # For LogisticRegression
import sklearn.neural_network    # For MLPClassifier
import torch
import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)  # Annoying
np.set_printoptions(precision=3, suppress=True)  # Print array values as 0.0023 instead of 2.352e-3