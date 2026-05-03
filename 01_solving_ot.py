# 01 Solving OT porblems with POT
# =========================      
# In this notebook, we will see how to solve optimal transport problems with the
# Python Optimal Transport (POT) library. We will start by importing the
# necessary libraries and defining our source and target distributions.


#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import ot

#%% Defining source and target distributions
# We will define two discrete distributions, one for the source and one for the
# target.

np.random.seed(0)  # For reproducibility

def get_source_and_target(n_s=100, n_t=None):
    if n_t is None:
        n_t = n_s
    # Source distribution: a Gausian distribution 
    
    x_source = np.random.randn(n_s, 2)
    
    # Target distribution: points on a circle

    angles = 2*np.pi*np.random.rand(n_t)
    x_target = 4*np.c_[np.cos(angles), np.sin(angles)] + 0.2*np.random.randn(n_t, 2)

    return x_source, x_target

x_s, x_t = get_source_and_target(n_s=100, n_t=150)

w_s = np.ones(x_s.shape[0]) / x_s.shape[0]  # Uniform weights for source
w_t = np.ones(x_t.shape[0]) / x_t.shape[0]  # Uniform weights for target

plt.figure(1, figsize=(6, 6))
plt.scatter(x_s[:, 0], x_s[:, 1], label='Source', alpha=0.5)
plt.scatter(x_t[:, 0], x_t[:, 1], label='Target', alpha=0.5)
plt.legend()
plt.title('Source and Target Distributions')
plt.show()


#%% Computing the cost matrix
# The cost matrix is a matrix that contains the cost of transporting mass from
# each point in the source distribution to each point in the target distribution.
# We will use the squared Euclidean distance as our cost function.

C = ot.dist(x_s, x_t, metric='euclidean')



