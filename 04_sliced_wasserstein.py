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

#%% computational time comparison

n_list = np.logspace(2, 4, 10, dtype=int)
sw_times = []
w_times = []

for n in n_list:
    x1, x2 = get_source_and_target(n_s=n, n_t=n, delta=delta)
    
    ot.tic()
    ot.sliced_wasserstein_distance(x1, x2, n_projections=100)
    sw_times.append(ot.toc())

    ot.tic()
    ot.solve_sample(x1, x2, metric='sqeuclidean').value**0.5
    w_times.append(ot.toc())

plt.figure(3, figsize=(6, 3))
plt.plot(n_list, sw_times, 'o-', label='Sliced Wasserstein Time')
plt.plot(n_list, w_times, 's-', label='Wasserstein Time')
plt.xlabel('Number of Points')
plt.ylabel('Time (s)')
plt.title('Sliced Wasserstein Computational Time')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()


#%% Computing the sliced Wasserstein barycenter with optimization

n_s = 50  # Number of source points
n_t = 100  # Number of target points

delta = 8

x1, x2 = get_source_and_target(n_s=n_s, n_t=n_t, delta=delta)
X1 = torch.tensor(x1, dtype=torch.float32)
X2 = torch.tensor(x2, dtype=torch.float32)

alpha = 0.5
n_bary = 50

X_init = torch.randn(n_bary, 2, requires_grad=True)

optimizer = torch.optim.Adam([X_init], lr=0.1)
n_iters = 100

n_projections = 100

losses = []

seed = 0

for i in range(n_iters):

    optimizer.zero_grad()
    
    loss1 = ot.sliced_wasserstein_distance(X_init, X1, n_projections=n_projections,seed=seed)**2
    loss2 = ot.sliced_wasserstein_distance(X_init, X2, n_projections=n_projections,seed=seed)**2

    loss = alpha*loss2 + (1-alpha)*loss1

    losses.append(loss.item())
    
    loss.backward()


    optimizer.step()

    if (i+1) % 10 == 0:
        print(f'Iteration {i+1}/{n_iters}, Sliced Wasserstein Distance: {loss.item():.4f}')

# plotting the loss curve
plt.figure(4, figsize=(6, 3))
plt.plot(losses)
plt.title('Sliced Wasserstein Barycenter Optimization')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid()
plt.show()

# plotting the barycenter
plt.figure(5, figsize=(6, 3))
plt.scatter(X1[:, 0].detach().numpy(), X1[:, 1].detach().numpy(), label='Distribution 1', alpha=0.5)
plt.scatter(X2[:, 0].detach().numpy(), X2[:, 1].detach().numpy(), label='Distribution 2', alpha=0.5)
plt.scatter(X_init[:, 0].detach().numpy(), X_init[:, 1].detach().numpy(), label='Sliced Wasserstein Barycenter', alpha=0.5, color='red')
plt.title('Sliced Wasserstein Barycenter')
plt.legend(loc='lower right')
plt.show()  


