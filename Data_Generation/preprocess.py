import sys
sys.path.append('C:\\UWMadisonResearch\\SGM_FNO_Closure')
import h5py
import torch
import torch.nn.functional as F
from utility import set_seed


##############################
#######  Data Loading ########
##############################
# Load raw data
device = torch.device('cuda')
filename = 'C:\\UWMadisonResearch\\SGM_FNO_Closure\\Data_Generation\\2d_ns_diffusion_40s_sto_midV_256.h5'
with h5py.File(filename, 'r') as file:
    sol_t = torch.tensor(file['t'][()], device='cuda')
    sol = torch.tensor(file['sol'][()], device='cuda')
    diffusion = torch.tensor(file['diffusion'][()], device='cuda')

# Only taking segments between 30s and 40s
sol_sliced_train = sol[:8, :, :, 3000:4001]
sol_sliced_test = sol[8:10, :, :, 3000:4001]
diffusion_sliced_train = diffusion[:8, :, :, 3000:4001]
diffusion_sliced_test = diffusion[8:10, :, :, 3000:4001]

sol_reshaped_train = sol_sliced_train.permute(0,3,1,2).reshape(-1, 256, 256)
diffusion_reshaped_train = diffusion_sliced_train.permute(0,3,1,2).reshape(-1, 256, 256)
sol_reshaped_test = sol_sliced_test.permute(0,3,1,2).reshape(-1, 256, 256)
diffusion_reshaped_test = diffusion_sliced_test.permute(0,3,1,2).reshape(-1, 256, 256)


set_seed(42)
indices = torch.randperm(sol_reshaped_train.shape[0])
sol_reshaped_train = sol_reshaped_train[indices]
diffusion_reshaped_train = diffusion_reshaped_train[indices]

set_seed(42)
indiced_test = torch.randperm(sol_reshaped_test.shape[0])
sol_reshaped_test = sol_reshaped_test[indiced_test]
diffusion_reshaped_test = diffusion_reshaped_test[indiced_test]

# Train/Test
Ntrain = 8000
Ntest = 2000

train_diffusion_256 = diffusion_reshaped_train[:Ntrain, :, :]
train_vorticity_256= sol_reshaped_train[:Ntrain, :, :]

test_diffusion_256 = diffusion_reshaped_test[:Ntest, :, :]
test_vorticity_256 = sol_reshaped_test[:Ntest, :, :]

# Downsampling
train_diffusion_64 = train_diffusion_256[:, ::4, ::4]
train_vorticity_64 = train_vorticity_256[:, ::4, ::4]

test_diffusion_64 = test_diffusion_256[:, ::4, ::4]
test_vorticity_64 = test_vorticity_256[:, ::4, ::4]

test_diffusion_128 = test_diffusion_256[:, ::2, ::2]
test_vorticity_128 = test_vorticity_256[:, ::2, ::2]

# Sparse information for interpolation conditioning
train_diffusion_64_sparse_interp = train_diffusion_64[:, ::4, ::4]
test_diffusion_64_sparse_interp = test_diffusion_64[:, ::4, ::4]
test_diffusion_128_sparse_interp = test_diffusion_128[:, ::8, ::8]
test_diffusion_256_sparse_interp = test_diffusion_256[:, ::16, ::16]



# Sparse information for convolution conditioning
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


filename = 'C:\\UWMadisonResearch\\SGM_FNO_Closure\\Data_Generation\\train_diffusion.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('train_diffusion_64', data=train_diffusion_64.cpu().numpy())
    file.create_dataset('train_vorticity_64', data=train_vorticity_64.cpu().numpy())
    file.create_dataset('train_diffusion_64_sparse_interp', data=train_diffusion_64_sparse_interp.cpu().numpy())
    file.create_dataset('train_diffusion_64_sparse_normalized', data=train_diffusion_64_sparse_normalized.cpu().numpy())

filename = 'C:\\UWMadisonResearch\\SGM_FNO_Closure\\Data_Generation\\test_diffusion.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('test_diffusion_64', data=test_diffusion_64.cpu().numpy())
    file.create_dataset('test_vorticity_64', data=test_vorticity_64.cpu().numpy())
    file.create_dataset('test_diffusion_128', data=test_diffusion_128.cpu().numpy())
    file.create_dataset('test_vorticity_128', data=test_vorticity_128.cpu().numpy())
    file.create_dataset('test_diffusion_256', data=test_diffusion_256.cpu().numpy())
    file.create_dataset('test_vorticity_256', data=test_vorticity_256.cpu().numpy())
    file.create_dataset('test_diffusion_64_sparse_interp', data=test_diffusion_64_sparse_interp.cpu().numpy())
    file.create_dataset('test_diffusion_128_sparse_interp', data=test_diffusion_128_sparse_interp.cpu().numpy())
    file.create_dataset('test_diffusion_256_sparse_interp', data=test_diffusion_256_sparse_interp.cpu().numpy())
    file.create_dataset('test_diffusion_64_sparse_normalized', data=test_diffusion_64_sparse_normalized.cpu().numpy())
    file.create_dataset('test_diffusion_128_sparse_normalized', data=test_diffusion_128_sparse_normalized.cpu().numpy())
    file.create_dataset('test_diffusion_256_sparse_normalized', data=test_diffusion_256_sparse_normalized.cpu().numpy())

