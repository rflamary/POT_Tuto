# -*- coding: utf-8 -*-
# 02_OT_Barycenters
# =================
# In this notebook, we will see how to compute Wasserstein barycenters with the
# POT library. A Wasserstein barycenter is a distribution that minimizes the
# weighted sum of Wasserstein distances to a set of input distributions. We will
# show how to compute both fixed support and free support barycenters.

# Author: Rémi Flamary
#
# License: MIT License

#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import ot

#%% Defining the distributions with fixed support

def get_distributions_fixed(n=100):

    a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T
    n_distributions = A.shape[1]

    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()

    return A, M, n_distributions

n = 100  # Number of points in the support
A, M, n_distributions = get_distributions_fixed(n=n)

# plotting the distributions
plt.figure(1, figsize=(6, 3))
for i in range(n_distributions):
    plt.plot(A[:, i], label=f'Distribution {i+1}')
plt.title('Input Distributions')
plt.legend()
plt.show()

#%% Computing the fixed support barycenter
# We will compute the barycenter of the two distributions with uniform weights.

# weights
alpha= 0.5
w = np.array([1-alpha, alpha])  # Uniform weights for the barycenter

bary_l2 = A.dot(w)

bary_ot = ot.bregman.barycenter(A, M, weights=w, reg=1e-3)

#bary_ot_exact = ot.lp.barycenter(A, M, weights=w)

# plotting the barycenters
plt.figure(1, figsize=(6, 3))
for i in range(n_distributions):
    plt.plot(A[:, i], label=f'Distribution {i+1}',alpha=0.5)
plt.plot(bary_l2, label='L2 Barycenter', linestyle='-')
plt.plot(bary_ot, label='OT Barycenter', linestyle='-')
# plt.plot(bary_ot_exact, label='OT Barycenter (Exact)', linestyle='--')
plt.title('Barycenters')
plt.legend()
plt.show()


#%% Defining the distributions with free support
# In this case, we will define the distributions as discrete distributions with
# free support. This means that the support of the distributions is not fixed and
# can be optimized during the barycenter computation.   

def get_source_and_target(n_s=100, n_t=None):
    if n_t is None:
        n_t = n_s
    # Source distribution: a Gausian distribution 
    
    x_source = np.random.randn(n_s, 2)*0.5
    
    # Target distribution: points on a circle

    angles = 2*np.pi*np.random.rand(n_t)
    x_target = 4*np.c_[np.cos(angles), np.sin(angles)] + 0.2*np.random.randn(n_t, 2)

    return x_source, x_target

n_s = 50  # Number of source points
n_t = 100  # Number of target points

x_1, x_2 = get_source_and_target(n_s=n_s, n_t=n_t)

lst_dist = [x_1, x_2]
lst_dist_weights = [ ot.utils.unif(x_1.shape[0]), ot.utils.unif(x_2.shape[0]) ] 

# plotting the distributions
plt.figure(1, figsize=(6, 6))
plt.scatter(x_1[:, 0], x_1[:, 1], label='Distribution 1', alpha=0.5)
plt.scatter(x_2[:, 0], x_2[:, 1], label='Distribution 2', alpha=0.5)
plt.title('Input Distributions')
plt.legend()
plt.show()  

#%% Computing the free support barycenter
# We will compute the barycenter of the two distributions with uniform weights.

# weights
alpha= 0.5
w = np.array([1-alpha, alpha])  # Uniform weights for the barycenter

n_bary = 50  # Number of points in the barycenter support
X_init = np.random.randn(n_bary, 2)  # Initial support for the barycenter

X_bary = ot.lp.free_support_barycenter(lst_dist, lst_dist_weights, X_init=X_init, weights=w)

#X_bary_reg = ot.bregman.free_support_sinkhorn_barycenter(lst_dist, lst_dist_weights, X_init=X_init, weights=w, reg=1e-1)

# plotting the barycenter
plt.figure(1, figsize=(6, 6))
plt.scatter(x_1[:, 0], x_1[:, 1], label='Distribution 1', alpha=0.2)
plt.scatter(x_2[:, 0], x_2[:, 1], label='Distribution 2', alpha=0.2)
plt.scatter(X_bary[:, 0], X_bary[:, 1], label='Barycenter', c='C3')
#plt.scatter(X_bary_reg[:, 0], X_bary_reg[:, 1], label='Barycenter (Regularized)', c='C4')
plt.title('Free Support Barycenter')
plt.legend()
plt.show()


