import sys
sys.path.append('C:\\UWMadisonResearch\\SGM_FNO_Closure\\DiffusionTerm_Generation')

import time
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
from torch.optim import Adam
from functools import partial
from tqdm import trange
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utility import (set_seed, energy_spectrum, get_sigmas_karras, sampler)

from Model_Designs import (marginal_prob_std, diffusion_coeff, FNO2d_Interp, loss_fn)

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
    train_diffusion_64_sparse_interp = torch.tensor(file['train_diffusion_64_sparse_interp'][:], device=device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_diffusion_64,
                                                                          train_vorticity_64,
                                                                          train_diffusion_64_sparse_interp),
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

model = FNO2d_Interp(marginal_prob_std_fn, modes, modes, width).cuda()
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
torch.save(model.state_dict(), 'SparseDiffusionModel_Interp.pth')







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




