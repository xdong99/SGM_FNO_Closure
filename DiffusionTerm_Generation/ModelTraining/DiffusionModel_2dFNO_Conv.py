import sys
sys.path.append('C:\\UWMadisonResearch\\SGM_FNO_Closure\\DiffusionTerm_Generation')

import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
from torch.optim import Adam
from functools import partial
from tqdm import trange

from Model_Designs import (marginal_prob_std, diffusion_coeff, FNO2d_Conv, loss_fn)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

# Load the data
train_file = 'C:\\UWMadisonResearch\\Conditional_Score_FNO\\Data_Generation\\train_diffusion.h5'
with h5py.File(train_file, 'r') as file:
    train_diffusion_64 = torch.tensor(file['train_diffusion_64'][:], device=device)
    train_vorticity_64 = torch.tensor(file['train_vorticity_64'][:], device=device)
    train_diffusion_64_sparse_normalized = torch.tensor(file['train_diffusion_64_sparse_normalized'][:], device=device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_diffusion_64,
                                                                          train_vorticity_64,
                                                                          train_diffusion_64_sparse_normalized),
                                                                          batch_size=200, shuffle=True)


################################
######## Model Training ########
################################
sigma = 26
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 8
width = 20
epochs = 1000
learning_rate = 0.001
scheduler_step = 200
scheduler_gamma = 0.5

model = FNO2d_Conv(marginal_prob_std_fn, modes, modes, width).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

tqdm_epoch = trange(epochs)

loss_history = []
rel_err_history = []

for epoch in tqdm_epoch:
    model.train()
    avg_loss = 0.
    num_items = 0
    rel_err = []
    for x, w, x_sparse in train_loader:
        x, w, x_sparse = x.cuda(), w.cuda(), x_sparse.cuda()
        optimizer.zero_grad()
        loss, score, real_score = loss_fn(model, x, w, x_sparse, marginal_prob_std_fn)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        relative_loss = torch.mean(torch.norm(score - real_score, 2, dim=(1, 2))
                                   / torch.norm(real_score, 2, dim=(1, 2)))
        rel_err.append(relative_loss.item())
    scheduler.step()
    avg_loss_epoch = avg_loss / num_items
    relative_loss_epoch = np.mean(rel_err)
    loss_history.append(avg_loss_epoch)
    rel_err_history.append(relative_loss_epoch)
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
torch.save(model.state_dict(), 'SparseDiffusionModel_Conv.pth')