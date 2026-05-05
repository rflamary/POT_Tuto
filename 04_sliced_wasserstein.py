# 04 Sliced Wasserstein Distance
# ==============================
# 

#%% importing libraries
import numpy as np
import matplotlib.pyplot as plt
import ot
import torch

#%% Defining the distributions



def get_source_and_target(n_s=100, n_t=None, delta=0):
    if n_t is None:
        n_t = n_s
    # Source distribution: a Gausian distribution 

    np.random.seed(0)
    
    x_source = np.random.randn(n_s, 2)*0.5
    
    # Target distribution: points on a circle

    angles = 2*np.pi*np.random.rand(n_t)
    x_target = 4*np.c_[np.cos(angles), np.sin(angles)] + 0.2*np.random.randn(n_t, 2)

    x_source[:, 0] -= delta/2
    x_target[:, 0] += delta/2

    return x_source, x_target

n_s = 50  # Number of source points
n_t = 100  # Number of target points

delta = 8
x1, x2 = get_source_and_target(n_s=n_s, n_t=n_t, delta=delta)


# plotting the distributions
plt.figure(1, figsize=(6, 3))
plt.scatter(x1[:, 0], x1[:, 1], label='Distribution 1', alpha=0.5)
plt.scatter(x2[:, 0], x2[:, 1], label='Distribution 2', alpha=0.5)
plt.title('Input Distributions')
plt.legend()
plt.show()


#%% Computing the sliced Wasserstein distance

lst_delta = np.linspace(0, 10, 20)
sw_distances = []
w_distances = []

for delta in lst_delta:
    x1, x2 = get_source_and_target(n_s=n_s, n_t=n_t, delta=delta)
    sw_dist = ot.sliced_wasserstein_distance(x1, x2, n_projections=100)
    sw_distances.append(sw_dist)

    w_dist = ot.solve_sample(x1, x2, metric='sqeuclidean').value**0.5
    w_distances.append(w_dist)

plt.figure(2, figsize=(6, 3))
plt.plot(lst_delta, sw_distances, 'o-', label='Sliced Wasserstein')
plt.plot(lst_delta, w_distances, 's-', label='Wasserstein')
plt.xlabel('Delta')
plt.ylabel('Distance')
plt.title('Distance vs. Shift')
pl.grid()
plt.legend()
plt.show()
