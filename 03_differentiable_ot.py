# 03 Differentiable Optimal Transport solvers
# ============================================
# In this notebook, we will explore how to use differentiable optimal transport
# solvers with the POT library. 

#%% Importing libraries
import numpy as np
import torch
import ot
import matplotlib.pyplot as plt



#%% Defining the distributions

np.random.seed(0)

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

X_init0, X_target = get_source_and_target(n_s=n_s, n_t=n_t)

X_init = torch.tensor(X_init0, dtype=torch.float32, requires_grad=True)
X_target = torch.tensor(X_target, dtype=torch.float32)

# plotting the distributions
plt.figure(1, figsize=(6, 6))
plt.scatter(X_init[:, 0].detach().numpy(), X_init[:, 1].detach().numpy(), label='Initial Distribution', alpha=0.5)
plt.scatter(X_target[:, 0].numpy(), X_target[:, 1].numpy(), label='Target Distribution', alpha=0.5)
plt.title('Distributions')
plt.legend(loc='upper right')
plt.show()

#%% peforming the gradient flow
# We will perform a gradient flow on the initial distribution to move it towards
# the target distribution.

n_iters = 100
lr =  5
reg = None
metric = 'euclidean'

X_init = torch.tensor(X_init0, dtype=torch.float32, requires_grad=True)

optimizer = torch.optim.SGD([X_init], lr=lr)

X_traj = np.zeros((n_iters, X_init.shape[0], X_init.shape[1]))
losses = np.zeros(n_iters)

for i in range(n_iters):

    X_traj[i] = X_init.detach().numpy()

    optimizer.zero_grad()
    # Compute the OT loss
    loss = ot.solve_sample(X_init, X_target, reg=reg, metric=metric).value

    losses[i] = loss.item()
    
    # Backpropagate the loss
    loss.backward()
    
    # Update the initial distribution
    optimizer.step()
    
    if (i+1) % 10 == 0:
        print(f'Iteration {i+1}/{n_iters}, Loss: {loss.item():.4f}')

# plot the loss curve
plt.figure(2, figsize=(6, 4))
plt.plot(losses)
plt.title('OT Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('OT Loss')
plt.grid()
plt.show()  

# plot the trajectory of the points
plt.figure(3, figsize=(6, 6))
plt.scatter(X_init0[:, 0], X_init0[:, 1], label='Initial Distribution', alpha=0.5, color='C0')
for i in range(X_init.shape[0]):
    plt.plot(X_traj[:, i, 0], X_traj[:, i, 1], alpha=0.5, color='C2')
plt.scatter(X_init[:, 0].detach().numpy(), X_init[:, 1].detach().numpy(), label='Final Distribution', color='C2')
plt.scatter(X_target[:, 0].numpy(), X_target[:, 1].numpy(), label='Target Distribution', alpha=0.5, color='C1')

plt.title('Trajectory of the points')
plt.legend(loc='upper right')
plt.show()

#%% optimizing the point weights


X_init = torch.tensor(X_init0, dtype=torch.float32)

w_unif = ot.utils.unif(X_init.shape[0])

W_init = torch.tensor(w_unif, dtype=torch.float32, requires_grad=True)

n_iters = 100
lr =  5
reg = None
metric = 'euclidean'

optimizer = torch.optim.SGD([W_init], lr=lr)

losses = np.zeros(n_iters)

for i in range(n_iters):

    optimizer.zero_grad()
    # Compute the OT loss

    w_iter = torch.softmax(W_init, dim=0)  # Ensure the weights are positive and sum to 1

    loss = ot.solve_sample(X_init, X_target, a=w_iter, reg=reg, metric=metric).value

    losses[i] = loss.item()
    
    # Backpropagate the loss
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    if (i+1) % 10 == 0:
        print(f'Iteration {i+1}/{n_iters}, Loss: {loss.item():.4f}')    

w_final = torch.softmax(W_init, dim=0).detach().numpy()

# plot the loss curve
plt.figure(4, figsize=(6, 4))
plt.plot(losses)
plt.title('OT Loss Curve (Optimizing Weights)')
plt.xlabel('Iteration')
plt.ylabel('OT Loss')
plt.grid()
plt.show()

# plot the final distribution with optimized weights
plt.figure(5, figsize=(6, 6))
plt.scatter(X_init[:, 0].detach().numpy(), X_init[:, 1].detach().numpy(), label='Initial Distribution', alpha=0.5, color='C0')
plt.scatter(X_init[:, 0].detach().numpy(), X_init[:, 1].detach().numpy(), s=w_final*1000, label='Optimized Distribution', color='C2', alpha=0.5)
plt.scatter(X_target[:, 0].numpy(), X_target[:, 1].numpy(), label='Target Distribution', alpha=0.5, color='C1')
plt.title('Final Distribution with Optimized Weights')
plt.legend(loc='upper right')
plt.show()

