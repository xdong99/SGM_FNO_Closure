import sys
sys.path.append('C:\\UWMadisonResearch\\Conditional_Score_FNO\\DiffusionTerm_Generation')

import time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
from torch.optim import Adam
from functools import partial
from tqdm import trange
import gc
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utility import (set_seed, marginal_prob_std, diffusion_coeff,FNO2d_Interp, FNO2d_Conv,
                     FNO2d_NoSparse, loss_fn, get_sigmas_karras, sampler)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")


#Convoluted Sparse Information
# Define the size of the convolutional kernel
kernel_size = 7
kernel64 = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
kernel64 = kernel64.to(device)

mask = torch.zeros_like(train_diffusion_64)
mask[:, ::4, ::4] = 1
train_diffusion_64_sparse = train_diffusion_64 * mask
test_diffusion_64_sparse = test_diffusion_64 * mask[:Ntest, :, :]

train_diffusion_64_sparse_squeezed = train_diffusion_64_sparse.unsqueeze(1)
train_diffusion_64_sparse_GF = F.conv2d(train_diffusion_64_sparse_squeezed, kernel64, padding='same')
train_diffusion_64_sparse_GF = train_diffusion_64_sparse_GF.squeeze(1)
train_diffusion_64_sparse_normalized = torch.empty_like(train_diffusion_64_sparse_GF)
for i in range(train_diffusion_64_sparse_GF.shape[0]):
    batch_sparse = train_diffusion_64_sparse[i][train_diffusion_64_sparse[i] != 0]
    batch_smoothed = train_diffusion_64_sparse_GF[i][train_diffusion_64_sparse_GF[i] != 0]
    sparse_min, sparse_max = torch.min(batch_sparse), torch.max(batch_sparse)
    smoothed_min, smoothed_max = torch.min(batch_smoothed), torch.max(batch_smoothed)
    batch_normalized = (train_diffusion_64_sparse_GF[i] - smoothed_min) / (smoothed_max - smoothed_min)
    batch_normalized = batch_normalized * (sparse_max - sparse_min) + sparse_min
    train_diffusion_64_sparse_normalized[i] = batch_normalized

test_diffusion_64_sparse_squeezed = test_diffusion_64_sparse.unsqueeze(1)
test_diffusion_64_sparse_GF = F.conv2d(test_diffusion_64_sparse_squeezed, kernel64, padding='same')
test_diffusion_64_sparse_GF = test_diffusion_64_sparse_GF.squeeze(1)
test_diffusion_64_sparse_normalized = torch.empty_like(test_diffusion_64_sparse_GF)
for i in range(test_diffusion_64_sparse_GF.shape[0]):
    batch_sparse = test_diffusion_64_sparse[i][test_diffusion_64_sparse[i] != 0]
    batch_smoothed = test_diffusion_64_sparse_GF[i][test_diffusion_64_sparse_GF[i] != 0]
    sparse_min, sparse_max = torch.min(batch_sparse), torch.max(batch_sparse)
    smoothed_min, smoothed_max = torch.min(batch_smoothed), torch.max(batch_smoothed)
    batch_normalized = (test_diffusion_64_sparse_GF[i] - smoothed_min) / (smoothed_max - smoothed_min)
    batch_normalized = batch_normalized * (sparse_max - sparse_min) + sparse_min
    test_diffusion_64_sparse_normalized[i] = batch_normalized

kernel_size = 15
kernel128 = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
kernel128 = kernel128.to(device)

mask_128 = torch.zeros_like(test_diffusion_128)
mask_128[:, ::8, ::8] = 1
test_diffusion_128_sparse = test_diffusion_128 * mask_128
test_diffusion_128_sparse_GF = F.conv2d(test_diffusion_128_sparse.unsqueeze(1), kernel128, padding='same')
test_diffusion_128_sparse_GF = test_diffusion_128_sparse_GF.squeeze(1)
test_diffusion_128_sparse_normalized = torch.empty_like(test_diffusion_128_sparse_GF)
for i in range(test_diffusion_128_sparse_GF.shape[0]):
    batch_sparse = test_diffusion_128_sparse[i][test_diffusion_128_sparse[i] != 0]
    batch_smoothed = test_diffusion_128_sparse_GF[i][test_diffusion_128_sparse_GF[i] != 0]
    sparse_min, sparse_max = torch.min(batch_sparse), torch.max(batch_sparse)
    smoothed_min, smoothed_max = torch.min(batch_smoothed), torch.max(batch_smoothed)
    batch_normalized = (test_diffusion_128_sparse_GF[i] - smoothed_min) / (smoothed_max - smoothed_min)
    batch_normalized = batch_normalized * (sparse_max - sparse_min) + sparse_min
    test_diffusion_128_sparse_normalized[i] = batch_normalized

kernel_size = 31
kernel256 = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
kernel256 = kernel256.to(device)

mask_256 = torch.zeros_like(test_diffusion_256)
mask_256[:, ::16, ::16] = 1
test_diffusion_256_sparse = test_diffusion_256 * mask_256
test_diffusion_256_sparse_GF = F.conv2d(test_diffusion_256_sparse.unsqueeze(1), kernel256, padding='same')
test_diffusion_256_sparse_GF = test_diffusion_256_sparse_GF.squeeze(1)
test_diffusion_256_sparse_normalized = torch.empty_like(test_diffusion_256_sparse_GF)
for i in range(test_diffusion_256_sparse_GF.shape[0]):
    batch_sparse = test_diffusion_256_sparse[i][test_diffusion_256_sparse[i] != 0]
    batch_smoothed = test_diffusion_256_sparse_GF[i][test_diffusion_256_sparse_GF[i] != 0]
    sparse_min, sparse_max = torch.min(batch_sparse), torch.max(batch_sparse)
    smoothed_min, smoothed_max = torch.min(batch_smoothed), torch.max(batch_smoothed)
    batch_normalized = (test_diffusion_256_sparse_GF[i] - smoothed_min) / (smoothed_max - smoothed_min)
    batch_normalized = batch_normalized * (sparse_max - sparse_min) + sparse_min
    test_diffusion_256_sparse_normalized[i] = batch_normalized

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
# scheduler_step = 200
# scheduler_gamma = 0.5

model = FNO2d_Conv(marginal_prob_std_fn, modes, modes, width).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

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
    # scheduler.step()
    avg_loss_epoch = avg_loss / num_items
    relative_loss_epoch = np.mean(rel_err)
    loss_history.append(avg_loss_epoch)
    rel_err_history.append(relative_loss_epoch)
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
torch.save(model.state_dict(), 'SparseDiffusionModelMidV_3040_Conv.pth')


################################
##########  Sampling ###########
################################

# define and load model
sigma = 26
modes = 8
width = 20
s = 64
sub = 8

marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

model = FNO2d_Interp(marginal_prob_std_fn, modes, modes, width).cuda()
ckpt = torch.load('C:\\UWMadisonResearch\\Conditional_Score_FNO\\DiffusionTerm_Generation'
                  '\\Trained_Models\\SparseDiffusionModelMidV_3040_interp_newer.pth', map_location=device)
model.load_state_dict(ckpt)

model = FNO2d_NoSparse(marginal_prob_std_fn, modes, modes, width).cuda()
ckpt = torch.load('C:\\UWMadisonResearch\\Conditional_Score_FNO\\DiffusionTerm_Generation'
                  '\\Trained_Models\\SparseDiffusionModelMidV_3040_nosparse.pth', map_location=device)
model.load_state_dict(ckpt)


model = FNO2d_Conv(marginal_prob_std_fn, modes, modes, width).cuda()
ckpt = torch.load('C:\\UWMadisonResearch\\Conditional_Score_FNO\\DiffusionTerm_Generation'
                  '\\Trained_Models\\SparseDiffusionModelMidV_3040_conv.pth', map_location=device)
model.load_state_dict(ckpt)


sde_time_data: float = 0.5
sde_time_min = 1e-3
sde_time_max = 0.1
steps = 10
time_noises = get_sigmas_karras(steps, sde_time_min, sde_time_max, device=device)


sample_batch_size = 100
sample_spatial_dim = 128
sample_device = torch.device('cuda')
num_steps = 10

sampler = partial(sampler,
                  score_model = model,
                    marginal_prob_std = marginal_prob_std_fn,
                    diffusion_coeff = diffusion_coeff_fn,
                    batch_size = sample_batch_size,
                    spatial_dim = sample_spatial_dim,
                    num_steps = num_steps,
                    time_noises = time_noises,
                    device = sample_device)

samples_64 = sampler(test_vorticity_64[:sample_batch_size, :, :], test_diffusion_64_sparse_normalized[:sample_batch_size, :, :])
samples_128 = sampler(test_vorticity_128[:sample_batch_size, :, :], test_diffusion_128_sparse_normalized[:sample_batch_size, :, :])
samples_256 = sampler(test_vorticity_256[:sample_batch_size, :, :], test_diffusion_256_sparse_normalized[:sample_batch_size, :, :])

samples_64_interp = sampler(test_vorticity_64[:sample_batch_size, :, :], test_diffusion_64_sparse[:sample_batch_size, :, :])
samples_128_interp = sampler(test_vorticity_128[:sample_batch_size, :, :], test_diffusion_128_sparse[:sample_batch_size, :, :])
samples_256_interp = sampler(test_vorticity_256[:sample_batch_size, :, :], test_diffusion_256_sparse[:sample_batch_size, :, :])

set_seed(12)

fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
plt.rcParams.update({'font.size': 16})

# Ticks setting
ticks_64 = np.arange(0, 64, 10 * 64 / 64)
ticks_64_y = np.arange(4, 65, 10 * 64 / 64)[::-1]
tick_labels_64 = [str(int(tick)) for tick in ticks_64]


ticks_128 = np.arange(0, 128, 10 * 128 / 64)
ticks_128_y = np.arange(8, 129, 10 * 128 / 64)[::-1]
tick_labels_128 = [str(int(tick)) for tick in ticks_128]

ticks_256 = np.arange(0, 256, 10 * 256 / 64)
ticks_256_y = np.arange(16, 257, 10 * 256 / 64)[::-1]
tick_labels_256 = [str(int(tick)) for tick in ticks_256]


# Assuming torch.manual_seed or equivalent has been set as needed
indices = [torch.randint(0, sample_batch_size, (1,)).item() for _ in range(4)]

# Plotting
for i, idx in enumerate(indices):
    j = i % 4

    # Truth plot 128
    data1 = test_diffusion_64[idx, ...].cpu().numpy()
    sns.heatmap(data1, ax=axs[0, j], cmap='rocket', cbar=(i % 4 == 3), vmin=-0.6, vmax=0.4)
    axs[0, j].set_title(f"Truth {j+1}")
    axs[0, j].set_xticks(ticks_64)
    axs[0, j].set_yticks(ticks_64_y)
    axs[0, j].set_xticklabels(tick_labels_64, rotation=0)
    axs[0, j].set_yticklabels(tick_labels_64, rotation=0)

    # Generated plot 128
    data2 = samples_64_interp[idx, ...].cpu().numpy()
    sns.heatmap(data2, ax=axs[1, j], cmap='rocket', cbar=(i % 4 == 3), vmin=-0.6, vmax=0.4)
    axs[1, j].set_title(f"Generated {j+1}")
    axs[1, j].set_xticks(ticks_64)
    axs[1, j].set_yticks(ticks_64_y)
    axs[1, j].set_xticklabels(tick_labels_64, rotation=0)
    axs[1, j].set_yticklabels(tick_labels_64)

    # Truth plot 256
    data3 = abs(samples_64_interp[idx, ...].cpu().numpy() - test_diffusion_64[idx, ...].cpu().numpy())
    sns.heatmap(data3, ax=axs[2, j], cmap='rocket', cbar=(i % 4 == 3), vmin=0, vmax=0.1)
    axs[2, j].set_title(f"Error {j+1}")
    axs[2, j].set_xticks(ticks_64)
    axs[2, j].set_yticks(ticks_64_y)
    axs[2, j].set_xticklabels(tick_labels_64, rotation=0)
    axs[2, j].set_yticklabels(tick_labels_64)

    # # Generated plot 256
    # data4 = test_diffusion_256[idx, ...].cpu().numpy()
    # sns.heatmap(data4, ax=axs[3, j], cmap='rocket', cbar=(i % 4 == 3), vmin=-0.6, vmax=0.4)
    # axs[3, j].set_title(f"Generated {j+1}")
    # axs[3, j].set_xticks(ticks_256)
    # axs[3, j].set_yticks(ticks_256)
    # axs[3, j].set_xticklabels(tick_labels_256, rotation=0)
    # axs[3, j].set_yticklabels(tick_labels_256)

for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust label size as needed
plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
plt.savefig('C:\\UWMadisonResearch\\Conditional_Score_FNO\\draft_plots\\G_test_64_interp.png', dpi=300, bbox_inches='tight')
plt.show()

nan_batches = torch.isnan(samples_64).any(dim=1).any(dim=1)
valid_indices = torch.where(~nan_batches)[0]
mse = torch.mean((samples_64[valid_indices] - test_diffusion_64[valid_indices, :, :])**2)
rel_mse = (torch.mean( torch.norm(samples_64[valid_indices] - test_diffusion_64[valid_indices, :, :], 2, dim=(1, 2))
                    / torch.norm(test_diffusion_64[valid_indices, :, :], 2, dim=(1, 2))) )
rel_mse_squ = (torch.mean( torch.norm(samples_64[valid_indices] - test_diffusion_64[valid_indices, :, :], 2, dim=(1, 2))**2
                    / torch.norm(test_diffusion_64[valid_indices, :, :], 2, dim=(1, 2))**2) )


mse = torch.mean((samples_128_interp[valid_indices] - test_diffusion_128[valid_indices, :, :])**2)
rel_mse = (torch.mean( torch.norm(samples_128_interp[valid_indices] - test_diffusion_128[valid_indices, :, :], 2, dim=(1, 2))
                    / torch.norm(test_diffusion_128[valid_indices, :, :], 2, dim=(1, 2))) )
rel_mse_squ = (torch.mean( torch.norm(samples_128_interp[valid_indices] - test_diffusion_128[valid_indices, :, :], 2, dim=(1, 2))**2
                    / torch.norm(test_diffusion_128[valid_indices, :, :], 2, dim=(1, 2))**2) )


nan_batches = torch.isnan(samples_256_interp).any(dim=1).any(dim=1)
valid_indices = torch.where(~nan_batches)[0]

mse = torch.mean((samples_256_interp[valid_indices] - test_diffusion_256[valid_indices, :, :])**2)
rel_mse = (torch.mean( torch.norm(samples_256_interp[valid_indices] - test_diffusion_256[valid_indices, :, :], 2, dim=(1, 2))
                    / torch.norm(test_diffusion_256[valid_indices, :, :], 2, dim=(1, 2))) )
rel_mse_squ = (torch.mean( torch.norm(samples_256_interp[valid_indices] - test_diffusion_256[valid_indices, :, :], 2, dim=(1, 2))**2
                    / torch.norm(test_diffusion_256[valid_indices, :, :], 2, dim=(1, 2))**2) )




def moving_average(data, window_size):
    """ Simple moving average """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def energy_spectrum(phi, lx=1, ly=1, smooth=True):
    # Assuming phi is of shape (time_steps, nx, ny)
    nx, ny = phi.shape[1], phi.shape[2]
    nt = nx * ny

    phi_h = np.fft.fftn(phi, axes=(1, 2)) / nt  # Fourier transform along spatial dimensions

    energy_h = 0.5 * (phi_h * np.conj(phi_h)).real  # Spectral energy density

    k0x = 2.0 * np.pi / lx
    k0y = 2.0 * np.pi / ly
    knorm = (k0x + k0y) / 3.0

    kxmax = nx // 2
    kymax = ny // 2

    wave_numbers = knorm * np.arange(0, nx)

    energy_spectrum = np.zeros(len(wave_numbers))

    for kx in range(nx):
        rkx = kx if kx <= kxmax else kx - nx
        for ky in range(ny):
            rky = ky if ky <= kymax else ky - ny
            rk = np.sqrt(rkx ** 2 + rky ** 2)
            k = int(np.round(rk))
            if k < len(wave_numbers):
                energy_spectrum[k] += np.sum(energy_h[:, kx, ky])

    energy_spectrum /= knorm

    if smooth:
        smoothed_spectrum = moving_average(energy_spectrum, 5)  # Smooth the spectrum
        smoothed_spectrum = np.append(smoothed_spectrum, np.zeros(4))  # Append zeros to match original length after convolution
        smoothed_spectrum[:4] = np.sum(energy_h[:, :4, :4].real, axis=(0, 1, 2)) / (knorm * phi.shape[0])  # First 4 values corrected
        energy_spectrum = smoothed_spectrum

    knyquist = knorm * min(nx, ny) / 2

    return knyquist, wave_numbers, energy_spectrum

k64_gauss, E64_gauss = energy_spectrum(samples_64.cpu())[1:]
k128_gauss, E128_gauss = energy_spectrum(samples_128.cpu())[1:]
k256_gauss, E256_gauss = energy_spectrum(samples_256.cpu())[1:]


k64_interp, E64_interp = energy_spectrum(samples_64_interp.cpu())[1:]
k128_interp, E128_interp = energy_spectrum(samples_128_interp.cpu())[1:]
k256_interp, E256_interp = energy_spectrum(samples_256_interp.cpu())[1:]

k_truth_64, E_truth_64 = energy_spectrum(test_diffusion_64[:sample_batch_size, :, :].cpu())[1:]
k_truth_128, E_truth_128 = energy_spectrum(test_diffusion_128[:sample_batch_size, :, :].cpu())[1:]
k_truth_256, E_truth_256 = energy_spectrum(test_diffusion_256[:sample_batch_size, :, :].cpu())[1:]

resolutions = [64, 128, 256]
gauss_kn = [k64_gauss, k128_gauss, k256_gauss]
gauss_E = [E64_gauss, E128_gauss, E256_gauss]

interp_kn = [k64_interp, k128_interp, k256_interp]
interp_E = [E64_interp, E128_interp, E256_interp]

truth_kn = [k_truth_64, k_truth_128, k_truth_256]
truth_E = [E_truth_64, E_truth_128, E_truth_256]

fig, axs = plt.subplots(1, 3, figsize=(28, 10), sharey=True)
fs = 34
plt.rcParams.update({'font.size': fs})
plt.rcParams.update({'legend.fontsize': 35})  # Ensure the legend font size is updated


for i, res in enumerate(resolutions):
    # Upper row plots
    col = i
    axs[col].loglog(truth_kn[i], truth_E[i], label='Truth', linestyle='-.', linewidth=4)
    axs[col].loglog(interp_kn[i], interp_E[i], label='Interpolation', linestyle=':', linewidth=4)
    axs[col].loglog(gauss_kn[i], gauss_E[i], label='Convolution', linestyle='--', linewidth=4)

    axs[col].set_ylim(1e-18, 5 * 1e-1)
    axs[col].set_xlim(0, 1e3)
    axs[col].set_xscale('log')
    axs[col].set_yscale('log')
    axs[col].set_title(f'Energy Spectrum of $G$ ({res}x{res})', fontsize = fs)
    axs[col].set_xlabel('Wave number (k)', fontsize = fs)

axs[0].set_ylabel('Energy (E)', fontsize=fs)

for ax in axs.flat:
    ax.set_ylim(1e-18, 5 * 1e-1)
    ax.set_xlim(0, 1e3)
    ax.set_xscale('log')
    ax.set_yscale('log')

# Create a shared legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=fs)
# Adjust the layout to make space for the legend
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust this value as needed
plt.savefig('C:\\UWMadisonResearch\\Conditional_Score_FNO\\draft_plots\\TKECompare.png', dpi=300,
            bbox_inches='tight')
plt.show()






### Do reverse SDE sampling every time step

filename = 'C:\\UWMadisonResearch\\Conditional_Score_FNO\\Data_Generation\\2d_ns_diffusion_50s_sto_midV.h5'

# Open the HDF5 file
with h5py.File(filename, 'r') as file:
    # Load data directly into PyTorch tensors on the specified device
    sol_t = torch.tensor(file['t'][()], device='cuda')
    sol = torch.tensor(file['sol'][()], device='cuda')
    diffusion = torch.tensor(file['diffusion'][()], device='cuda')
    nonlinear = torch.tensor(file['nonlinear'][()], device='cuda')

## Vorticity Generation
import math
delta_t = 1e-3
nu = 1e-3
shifter = 30000
sample_size = 1
num_steps = 1000
total_steps = 20000
s = 64

vorticity = sol[7:7+sample_size, :, :, shifter]
vorticity_series = torch.zeros(sample_size, s, s, total_steps)
vorticity_NoG = torch.zeros(sample_size, s, s, total_steps)

t = torch.linspace(0, 1, s + 1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))




# Define the size of the convolutional kernel

diffusion_target = diffusion[7:7+sample_size, :, :, shifter: shifter+total_steps]
vorticity_condition = sol[7:7+sample_size, :, :, shifter: shifter+total_steps]
mask = torch.zeros_like(diffusion_target)
mask[:, ::4, ::4, :] = 1
diffusion_target_sparse = diffusion_target * mask
diffusion_target_sparse_GF = torch.empty_like(diffusion_target_sparse)

for t in range(total_steps):
    slice_squeezed = diffusion_target_sparse[:, :, :, t].unsqueeze(1)
    slice_convolved  = F.conv2d(slice_squeezed, kernel64, padding='same')
    diffusion_target_sparse_GF[:, :, :, t] = slice_convolved.squeeze(1)

diffusion_target_sparse_normalized = torch.empty_like(diffusion_target_sparse_GF)
for i in range(diffusion_target_sparse_GF.shape[0]):
    for t in range(diffusion_target_sparse_GF.shape[3]):
        batch_sparse = diffusion_target_sparse[i, :, :, t][diffusion_target_sparse[i, :, :, t] != 0]
        batch_smoothed = diffusion_target_sparse_GF[i, :, :, t][diffusion_target_sparse_GF[i, :, :, t] != 0]
        sparse_min, sparse_max = torch.min(batch_sparse), torch.max(batch_sparse)
        smoothed_min, smoothed_max = torch.min(batch_smoothed), torch.max(batch_smoothed)
        batch_normalized = (diffusion_target_sparse_GF[i, :, :, t] - smoothed_min) / (smoothed_max - smoothed_min)
        batch_normalized = batch_normalized * (sparse_max - sparse_min) + sparse_min
        diffusion_target_sparse_normalized[i, :, :, t] = batch_normalized


### Do reverse SDE sampling every 5 time steps

start_time = time.time()
# Do one Cranck-Nicolson step
N1, N2 = vorticity.size()[-2], vorticity.size()[-1]

# Maximum frequency
k_max1 = math.floor(N1 / 2.0)
k_max2 = math.floor(N1 / 2.0)

# Wavenumbers in y-direction
k_y = torch.cat((torch.arange(start=0, end=k_max2, step=1, device=vorticity.device),
                 torch.arange(start=-k_max2, end=0, step=1, device=vorticity.device)), 0).repeat(N1, 1).transpose(0, 1)
# Wavenumbers in x-direction
k_x = torch.cat((torch.arange(start=0, end=k_max1, step=1, device=vorticity.device),
                 torch.arange(start=-k_max1, end=0, step=1, device=vorticity.device)), 0).repeat(N2, 1)
# Negative Laplacian in Fourier space
lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
lap[0, 0] = 1.0

# Dealiasing mask
dealias = torch.unsqueeze(
    torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max2, torch.abs(k_x) <= (2.0 / 3.0) * k_max1).float(), 0)

# Initial vorticity to Fourier space
w_h = torch.fft.fftn(vorticity, dim=[1, 2])
w_h = torch.stack([w_h.real, w_h.imag], dim=-1)

# forcing
f_h = torch.fft.fftn(f, dim=[-2, -1])
f_h = torch.stack([f_h.real, f_h.imag], dim=-1)
# If same forcing for the whole batch
if len(f_h.size()) < len(w_h.size()):
    f_h = torch.unsqueeze(f_h, 0)



for i in range(total_steps):
    print(i)
    psi_h = w_h.clone()
    psi_h[..., 0] = psi_h[..., 0] / lap
    psi_h[..., 1] = psi_h[..., 1] / lap

    # Velocity field in x-direction = psi_y
    q = psi_h.clone()
    temp = q[..., 0].clone()
    q[..., 0] = -2 * math.pi * k_y * q[..., 1]
    q[..., 1] = 2 * math.pi * k_y * temp
    q = torch.fft.ifftn(torch.view_as_complex(q), dim=[1, 2], s=(N1, N2)).real

    # Velocity field in y-direction = -psi_x
    v = psi_h.clone()
    temp = v[..., 0].clone()
    v[..., 0] = 2 * math.pi * k_x * v[..., 1]
    v[..., 1] = -2 * math.pi * k_x * temp
    v = torch.fft.ifftn(torch.view_as_complex(v), dim=[1, 2], s=(N1, N2)).real

    # Partial x of vorticity
    w_x = w_h.clone()
    temp = w_x[..., 0].clone()
    w_x[..., 0] = -2 * math.pi * k_x * w_x[..., 1]
    w_x[..., 1] = 2 * math.pi * k_x * temp
    w_x = torch.fft.ifftn(torch.view_as_complex(w_x), dim=[1, 2], s=(N1, N2)).real

    # Partial y of vorticity
    w_y = w_h.clone()
    temp = w_y[..., 0].clone()
    w_y[..., 0] = -2 * math.pi * k_y * w_y[..., 1]
    w_y[..., 1] = 2 * math.pi * k_y * temp
    w_y = torch.fft.ifftn(torch.view_as_complex(w_y), dim=[1, 2], s=(N1, N2)).real

    F_h = torch.fft.fftn(q * w_x + v * w_y, dim=[1, 2])
    F_h = torch.stack([F_h.real, F_h.imag], dim=-1)

    # Dealias
    F_h[..., 0] = dealias * F_h[..., 0]
    F_h[..., 1] = dealias * F_h[..., 1]

    diffusion_target_sparse_normalized_iter = diffusion_target_sparse_normalized[:, :, :, i]
    diffusion_target = diffusion[7:7+sample_size, :, :, shifter+i]

    # if i % 5 == 0:
    #     sampler = partial(sampler,
    #                       score_model=model,
    #                       marginal_prob_std=marginal_prob_std_fn,
    #                       diffusion_coeff=diffusion_coeff_fn,
    #                       batch_size=1,
    #                       spatial_dim=64,
    #                       num_steps=10,
    #                       device='cuda')
    #
    #     diffusion_sample = sampler(vorticity, diffusion_target_sparse_normalized_iter)
    #     mse = torch.mean((diffusion_sample - diffusion_target)**2)
    #     rmse = torch.mean(torch.norm(diffusion_sample - diffusion_target, 2, dim=(1, 2)) / torch.norm(diffusion_target, 2, dim=(1, 2)))
    #     print(f"MSE: {mse:.8f}, RMSE: {rmse:.8f}")
    #
    # else:
    #     diffusion_sample = diffusion_sample + torch.randn_like(diffusion_sample) * 0.00005
    #
    # # laplacian term
    # diffusion_h = torch.fft.fftn(diffusion_sample, dim=[1, 2])
    # diffusion_h = torch.stack([diffusion_h.real, diffusion_h.imag], dim=-1)
    #
    # w_h[..., 0] = ((w_h[..., 0] - delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + 0.5 * delta_t * diffusion_h[..., 0])
    #                / (1.0 + 0.5 * delta_t * nu * lap))
    # w_h[..., 1] = ((w_h[..., 1] - delta_t * F_h[..., 1] + delta_t * f_h[..., 1] + 0.5 * delta_t * diffusion_h[..., 1])
    #                / (1.0 + 0.5 * delta_t * nu * lap))
    #
    # vorticity_series[..., i] = vorticity

    w_h[..., 0] = ((w_h[..., 0] - delta_t * F_h[..., 0] + delta_t * f_h[..., 0])
                   / (1.0 + 0.5 * delta_t * nu * lap))
    w_h[..., 1] = ((w_h[..., 1] - delta_t * F_h[..., 1] + delta_t * f_h[..., 1])
                   / (1.0 + 0.5 * delta_t * nu * lap))

    vorticity_NoG[..., i] = vorticity


    vorticity = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1, 2], s=(N1, N2)).real

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")




def relative_mse(tensor1, tensor2):
    """Calculate the Relative Mean Squared Error between two tensors."""
    rel_mse = torch.mean(torch.norm(tensor1 - tensor2, 2, dim=(0, 1)) / torch.norm(tensor2, 2, dim=(0, 1)))
    return rel_mse

def cal_mse(tensor1, tensor2):
    """Calculate the Mean Squared Error between two tensors."""
    mse = torch.mean((tensor1 - tensor2)**2)
    return mse
def calculate_pattern_correlation(tensor_a, tensor_b):
    pattern_correlations = []
    for i in range(tensor_a.shape[0]):  # Iterate over the batch dimension
        a = tensor_a[i].flatten()  # Flatten spatial dimensions
        b = tensor_b[i].flatten()
        mean_a = a.mean()
        mean_b = b.mean()
        a_centered = a - mean_a
        b_centered = b - mean_b
        covariance = (a_centered * b_centered).sum()
        std_a = torch.sqrt((a_centered**2).sum())
        std_b = torch.sqrt((b_centered**2).sum())
        correlation = covariance / (std_a * std_b)
        pattern_correlations.append(correlation)
    return torch.tensor(pattern_correlations)


# Assuming 'vorticity_series', 'vorticity_NoG', and 'sol' are preloaded tensors
shifter = 30000
k = 0
fs = 34
# Create a figure and a grid of subplots
fig, axs = plt.subplots(5, 5, figsize=(30, 32), gridspec_kw={'width_ratios': [1]*4 + [1.073]})

# Plot each row using seaborn heatmap
for row in range(5):
    for i in range(5):  # Loop through all ten columns
        ax = axs[row, i]

        j = i * 4999
        generated = vorticity_series[k, :, :, j].cpu()
        generated_nog = vorticity_NoG[k, :, :, j].cpu()
        truth = sol[k + 7, :, :, shifter + j].cpu()
        error_field = abs(generated - truth)
        error_field_nog = abs(generated_nog - truth)

        rmse = relative_mse(torch.tensor(generated), torch.tensor(truth)).item()
        mse = cal_mse(torch.tensor(generated), torch.tensor(truth)).item()
        pc_mean = calculate_pattern_correlation(torch.tensor(generated), torch.tensor(truth)).mean().item()

        rmse_nog = relative_mse(torch.tensor(generated_nog), torch.tensor(truth)).item()
        mse_nog = cal_mse(torch.tensor(generated_nog), torch.tensor(truth)).item()
        pc_nog_mean = calculate_pattern_correlation(torch.tensor(generated_nog), torch.tensor(truth)).mean().item()

        print(f"Time: {sol_t[shifter + j]:.2f}s")
        print(f"RMSE: {rmse:.4f}, Pattern Correlation: {pc_mean:.4f}")
        print(f"MSE: {mse:.8f}")
        print(f"RMSE NoG: {rmse_nog:.4f}, Pattern Correlation NoG: {pc_nog_mean:.4f}")
        print(f"MSE NoG: {mse_nog:.4f}")

        # Set individual vmin and vmax based on the row
        if row == 0:
            data = truth
            vmin, vmax = -2.0, 2.0  # Limits for Truth and Generated rows
            ax.set_title(f't = {sol_t[shifter + j]:.2f}s', fontsize=fs)
        elif row == 1:
            data = generated
            vmin, vmax = -2.0, 2.0  # Limits for Truth and Generated rows
        elif row == 2:
            data = generated_nog
            vmin, vmax = -2.0, 2.0
        elif row == 3:
            data = error_field
            vmin, vmax = 0, 1.5
        else:
            data = error_field_nog
            vmin, vmax = 0, 1.5
        # Plot heatmap
        sns.heatmap(data, ax=ax, cmap="rocket", vmin=vmin, vmax=vmax, square=True, cbar=False)

        ax.axis('off')  # Turn off axis for cleaner look

        if i == 4:
            # Create a new axis for the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(ax.collections[0], cax=cax, ticks=np.linspace(vmin, vmax, 5))
            cax.tick_params(labelsize=fs)

            # Format tick labels based on the row
            if row < 3:  # For the first two rows
                cb.ax.set_yticklabels(['{:.1f}'.format(tick) for tick in np.linspace(vmin, vmax, 5)])
            else:  # For the last row
                cb.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in np.linspace(vmin, vmax, 5)])

# Add row titles on the side
row_titles = ['Truth', 'Simulation with G', 'Simulation w/o G', 'Error with G', 'Error w/o G']
for ax, row_title in zip(axs[:, 0], row_titles):
    ax.annotate(row_title, xy=(0.1, 0.5), xytext=(-50, 0),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='center', rotation=90, fontsize=fs)

plt.tight_layout()  # Adjust the subplots to fit into the figure area
plt.savefig('C:\\UWMadisonResearch\\Conditional_Score_FNO\\draft_plots\\surrogate_64', dpi=300,
                                                                    bbox_inches='tight')
plt.show()








import matplotlib.gridspec as gridspec

# Time values in seconds for the x-axis
time_values = [30, 35, 40, 45, 50]

# MSE and RMSE data for simulations
sim_vort_mse_I = [0, 5.9232e-02, 3.0145e-01, 7.1893e-01, 8.5814e-01]
sim_vort_rmse_I = [0, 0.2431, 0.5771, 0.9246, 1.0315]
sim_vort_mse_II = [0, 1.2402e-04 ,1.3887e-03, 3.3429e-03, 3.1497e-03]
sim_vort_rmse_II = [0, 0.0111 ,0.0392, 0.0631, 0.0625]
# sim_vort_mse_III = [0, 2.0049e-04, 2.1207e-03, 4.8836e-03, 4.4227e-03]
# sim_vort_rmse_III = [0, 0.0141, 0.0484, 0.0762, 0.0741]
sim_vort_mse_III = [0, 1.3740e-04       , 1.5141e-03       , 3.8441e-03       , 3.6798e-03]
sim_vort_rmse_III = [0, 0.0117           , 0.0409           , 0.0676           , 0.0675]
sim_vort_mse_IV = [0, 1.2028e-04, 1.2698e-03 ,2.7910e-03 ,2.7792e-03]
sim_vort_rmse_IV = [0, 0.0110, 0.0375, 0.0576, 0.0587]

# Create a figure with a custom gridspec layout
fig = plt.figure(figsize=(28, 10))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
fs = 35
# MSE Plot
ax0 = plt.subplot(gs[0])
ax0.plot(time_values, sim_vort_mse_II, marker='o', linestyle=":", markersize=10, linewidth=4, label="Simulation II")
ax0.plot(time_values, sim_vort_mse_III, marker='o', linestyle="--", markersize=10, linewidth=4, label="Simulation III")
ax0.plot(time_values, sim_vort_mse_IV, marker='o', linestyle="-.", markersize=10, linewidth=4, label="Simulation IV")
ax0.set_title("$D_{\\text{MSE}}$ Comparison", fontsize=fs)
ax0.set_xlabel("Time (s)", fontsize=fs)
ax0.set_ylabel("$D_{\\text{MSE}}$", fontsize=fs)
ax0.set_yticks([0.000, 0.001, 0.002, 0.003, 0.004])
ax0.tick_params(axis='both', which='major', labelsize=fs)

# RMSE Plot
ax1 = plt.subplot(gs[1])
ax1.plot(time_values, sim_vort_rmse_II, marker='o', linestyle=":", markersize=10, linewidth=4, label="Simulation II")
ax1.plot(time_values, sim_vort_rmse_III, marker='o', linestyle="--", markersize=10, linewidth=4, label="Simulation III")
ax1.plot(time_values, sim_vort_rmse_IV, marker='o', linestyle="-.", markersize=10, linewidth=4, label="Simulation IV")
ax1.set_title("$D_{\\text{RE}}$ Comparison", fontsize=fs)
ax1.set_xlabel("Time (s)", fontsize=fs)
ax1.set_ylabel("$D_{\\text{RE}}$", fontsize=fs)
ax1.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08])
ax1.tick_params(axis='both', which='major', labelsize=fs)

# Create a shared legend
handles, labels = ax0.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=fs)
plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjust this value as needed
plt.savefig('C:\\UWMadisonResearch\\Conditional_Score_FNO\\draft_plots\\MSE_RE_Comparison_new', dpi=300)
plt.show()



import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation


fig = plt.figure(figsize=(40, 30))  # Increased height slightly
wid = 0.26
hei = 0.35  # Reduced height slightly to make room for title and metrics
gap = 0.01  # Gap between plots

# Define the positions for each subplot [left, bottom, width, height]
ax1_pos = [0.02, 0.32, wid, hei]  # Truth (centered vertically)
ax2_pos = [0.35, 0.52, wid, hei]  # Generated with G (top right)
ax3_pos = [0.35, 0.1, wid, hei]  # Generated w/o G (bottom right)
ax4_pos = [0.68, 0.52, wid, hei]  # Error with G (top far right)
ax5_pos = [0.68, 0.1, wid, hei]  # Error w/o G (bottom far right)

# Create the subplots with the defined positions
ax1 = fig.add_axes(ax1_pos)
ax2 = fig.add_axes(ax2_pos)
ax3 = fig.add_axes(ax3_pos)
ax4 = fig.add_axes(ax4_pos)
ax5 = fig.add_axes(ax5_pos)

import seaborn as sns
import numpy as np

# Ticks setting
ticks_64 = np.arange(0, 64, 10 * 64 / 64)
ticks_64_y = np.arange(4, 65, 10 * 64 / 64)[::-1]
tick_labels_64 = [str(int(tick)) for tick in ticks_64]

def animate(t):
    global sol, vorticity_series, vorticity_NoG, sol_t
    fs = 30
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.cla()

    if hasattr(animate, 'colorbar_axes'):
        for cax in animate.colorbar_axes:
            cax.remove()

    if hasattr(animate, 'txt'):
        for txt in animate.txt:
            txt.remove()

    frame_index = min(10 * t, 19999)

    # Define color limits
    vorticity_limits = [-2.0, 2.0]
    error_limits = [0.00, 1.50]

    # Plot for sol tensor (Truth)
    sns.heatmap(sol[k + 7, ..., shifter + 10 * t].cpu().detach(), ax=ax1, cmap='rocket',
                cbar_ax=fig.add_axes([ax1_pos[0] + ax1_pos[2] + gap, ax1_pos[1], 0.01, ax1_pos[3]]),
                vmin=vorticity_limits[0], vmax=vorticity_limits[-1])
    ax1.set_title("Truth", fontsize=fs)
    ax1.collections[0].colorbar.set_ticks(np.linspace(vorticity_limits[0], vorticity_limits[-1], 5))
    ax1.collections[0].colorbar.ax.tick_params(labelsize=fs)

    # Plot for vorticity_series tensor (Generated with G)
    sns.heatmap(vorticity_series[k, ..., frame_index].cpu().detach(), ax=ax2, cmap='rocket',
                cbar_ax=fig.add_axes([ax2_pos[0] + ax2_pos[2] + gap, ax2_pos[1], 0.01, ax2_pos[3]]),
                vmin=vorticity_limits[0], vmax=vorticity_limits[-1])
    ax2.set_title("Generated with G", fontsize=fs)
    ax2.collections[0].colorbar.set_ticks(np.linspace(vorticity_limits[0], vorticity_limits[-1], 5))
    ax2.collections[0].colorbar.ax.tick_params(labelsize=fs)

    # Plot for vorticity_NoG tensor (Generated w/o G)
    sns.heatmap(vorticity_NoG[k, ..., frame_index].cpu().detach(), ax=ax3, cmap='rocket',
                cbar_ax=fig.add_axes([ax3_pos[0] + ax3_pos[2] + gap, ax3_pos[1], 0.01, ax3_pos[3]]),
                vmin=vorticity_limits[0], vmax=vorticity_limits[-1])
    ax3.set_title("Generated w/o G", fontsize=fs)
    ax3.collections[0].colorbar.set_ticks(np.linspace(vorticity_limits[0], vorticity_limits[-1], 5))
    ax3.collections[0].colorbar.ax.tick_params(labelsize=fs)

    # Calculate and plot the absolute error with G
    abs_error_2 = torch.abs(sol[k + 7, ..., shifter + 10 * t].cpu() - vorticity_series[k, ..., frame_index].cpu())
    sns.heatmap(abs_error_2, ax=ax4, cmap='rocket',
                cbar_ax=fig.add_axes([ax4_pos[0] + ax4_pos[2] + gap, ax4_pos[1], 0.01, ax4_pos[3]]),
                vmin=error_limits[0], vmax=error_limits[-1])
    ax4.set_title("Error with G", fontsize=fs)
    ax4.collections[0].colorbar.set_ticks(np.linspace(error_limits[0], error_limits[-1], 5))
    ax4.collections[0].colorbar.ax.tick_params(labelsize=fs)

    # Calculate and plot the absolute error without G
    abs_error = torch.abs(sol[k + 7, ..., shifter + 10 * t].cpu() - vorticity_NoG[k, ..., frame_index].cpu())
    sns.heatmap(abs_error, ax=ax5, cmap='rocket',
                cbar_ax=fig.add_axes([ax5_pos[0] + ax5_pos[2] + gap, ax5_pos[1], 0.01, ax5_pos[3]]),
                vmin=error_limits[0], vmax=error_limits[-1])
    ax5.set_title("Error w/o G", fontsize=fs)
    ax5.collections[0].colorbar.set_ticks(np.linspace(error_limits[0], error_limits[-1], 5))
    ax5.collections[0].colorbar.ax.tick_params(labelsize=fs)

    animate.colorbar_axes = [ax1.collections[0].colorbar.ax, ax2.collections[0].colorbar.ax,
                             ax3.collections[0].colorbar.ax, ax4.collections[0].colorbar.ax,
                             ax5.collections[0].colorbar.ax]

    # Calculate metrics
    re_with_G = relative_mse(sol[k + 7, ..., shifter + 10 * t].cpu(), vorticity_series[k, ..., frame_index].cpu())
    re_without_G = relative_mse(sol[k + 7, ..., shifter + 10 * t].cpu(), vorticity_NoG[k, ..., frame_index].cpu())
    mse_with_G = cal_mse(sol[k + 7, ..., shifter + 10 * t].cpu(), vorticity_series[k, ..., frame_index].cpu())
    mse_without_G = cal_mse(sol[k + 7, ..., shifter + 10 * t].cpu(), vorticity_NoG[k, ..., frame_index].cpu())

    # Update figure title and captions with metrics
    fig.suptitle(r'$\nu = 10^{-3}, \beta = 0.00005, dt = 10^{-3}$' + '\n' + f't = {sol_t[shifter + 10 * t].item():.3f}',
                 fontsize=fs, y=0.95)

    txt1 = fig.text(0.65, 0.48,
                    f'MSE (with G): {mse_with_G.item():.4f}, RE (with G): {re_with_G.mean().item():.4f}',
                    ha='center', fontsize=fs)
    txt2 = fig.text(0.65, 0.06,
                    f'MSE (w/o G): {mse_without_G.item():.4f}, RE (w/o G): {re_without_G.mean().item():.4f}',
                    ha='center', fontsize=fs)

    animate.txt = [txt1, txt2]

    # Adjust x and y axis ticks
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xticks(ticks_64)
        ax.set_yticks(ticks_64_y)
        ax.tick_params(axis='both', which='major', labelsize=fs, rotation=0)
        ax.set_xticklabels(tick_labels_64, rotation=0, ha='center')
        ax.set_yticklabels(tick_labels_64, rotation=0, va='center')

    # Print progress
    progress = (t + 1) / 2000 * 100
    if t % 10 == 0:
        print(f"Progress: {progress:.2f}%")


# Create the animation
Animation1 = matplotlib.animation.FuncAnimation(fig, animate, frames=2000)
plt.close(fig)  # This prevents the static plot from displaying in Jupyter notebooks

# Save the animation
Animation1.save('Animation3050WithAndWithoutG.mp4', writer='ffmpeg', fps=60)




