# 01 Solving OT porblems with POT
# =========================      
# In this notebook, we will see how to solve optimal transport problems with the
# Python Optimal Transport (POT) library. We will start by importing the
# necessary libraries and defining our source and target distributions.


#%% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.plot

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

n_s = 50  # Number of source points
n_t = 100  # Number of target points

x_s, x_t = get_source_and_target(n_s=n_s, n_t=n_t)

unif_weights = True  # We will use uniform weights for both distributions

if unif_weights:

    w_s = np.ones(x_s.shape[0]) / x_s.shape[0]  # Uniform weights for source
    w_t = np.ones(x_t.shape[0]) / x_t.shape[0]  # Uniform weights for target

else:
    w_s = np.random.rand(x_s.shape[0])
    w_s /= w_s.sum()  # Normalize to sum to 1

    w_t = np.random.rand(x_t.shape[0])
    w_t /= w_t.sum()  # Normalize to sum to 1

# Let's visualize the source and target distributions.
scale = 5000
plt.figure(1, figsize=(6, 6))
plt.scatter(x_s[:, 0], x_s[:, 1], s=w_s*scale, label='Source', alpha=0.5)
plt.scatter(x_t[:, 0], x_t[:, 1], s=w_t*scale, label='Target', alpha=0.5)
plt.legend()
plt.title('Source and Target Distributions')
plt.show()


#%% Computing the cost matrix
# The cost matrix is a matrix that contains the cost of transporting mass from
# each point in the source distribution to each point in the target distribution.
# We will use the squared Euclidean distance as our cost function.

C = ot.dist(x_s, x_t, metric='euclidean')

# let us look at the cost matrix
plt.figure(2, figsize=(6, 4))
plt.imshow(C, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Cost Matrix')
plt.xlabel('Target Points')
plt.ylabel('Source Points')
plt.show()

#%% Solving the optimal transport problem
# We will use the `ot.solve` function to solve the optimal transport problem.
# This function takes the cost matrix and the weights of the source and target
# distributions as input and returns the optimal transport plan.

# Solving the optimal transport problem
res = ot.solve(C, w_s, w_t)

ot_value = res.value
print(f'Optimal transport cost: {ot_value:.4f}')
T = res.plan

# Visualizing the transport plan
# The transport plan is a matrix that indicates how much mass is transported from
# each point in the source distribution to each point in the target
# distribution.

def plot_transport_plan(x_s, x_t, T):
    plt.figure(figsize=(6, 6))
    ot.plot.plot2D_samples_mat(x_s, x_t, T, c='k', alpha=0.5)
    plt.scatter(x_s[:, 0], x_s[:, 1], s=w_s*scale, label='Source', zorder=2, edgecolors='k')
    plt.scatter(x_t[:, 0], x_t[:, 1], s=w_t*scale, label='Target', zorder=2, edgecolors='k')
    plt.legend()
    plt.title('Optimal Transport Plan')
    plt.show()

plot_transport_plan(x_s, x_t, T)


#%% Solving the empirical OT problem
# the cost matrix does not need to be computed explicitly, we can directly solve
# the # empirical OT problem using the `ot.solve_sample` function, which takes the
# weights and the samples of the source and target distributions as input.

res_emp = ot.solve_sample(x_s, x_t, metric='euclidean')

ot_value_emp = res_emp.value
print(f'Empirical optimal transport cost: {ot_value_emp:.4f}')
T_emp = res_emp.plan    

plot_transport_plan(x_s, x_t, T_emp)

#%% Solving the entropic regularized OT problem
# The entropic regularized OT problem adds an entropy term to the objective
# function, which encourages the transport plan to be more spread out. 

reg = 0.01  # Regularization parameter
res_entropic = ot.solve_sample(x_s, x_t, metric='euclidean', reg=reg)
ot_value_entropic = res_entropic.value
print(f'Entropic regularized optimal transport cost: {ot_value_entropic:.4f}')
T_entropic = res_entropic.plan

plot_transport_plan(x_s, x_t, T_entropic)




