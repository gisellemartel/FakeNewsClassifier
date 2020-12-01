#Library Imports for item-based collaborative filter:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Library imports for deep-learning recommender
import sklearn
import sklearn.preprocessing     # For StandardScaler
import sklearn.linear_model      # For LogisticRegression
import sklearn.neural_network    # For MLPClassifier
import torch
import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)  # Annoying
np.set_printoptions(precision=3, suppress=True)  # Print array values as 0.0023 instead of 2.352e-3