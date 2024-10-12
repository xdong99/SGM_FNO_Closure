import sys
sys.path.append('C:\\UWMadisonResearch\\Conditional_Score_FNO\\DiffusionTerm_Generation')
import h5py
import torch
from utility import set_seed


##############################
#######  Data Loading ########
##############################
# Load raw data
device = torch.device('cuda')
filename = 'C:\\UWMadisonResearch\\Conditional_Score_FNO\\Data_Generation\\2d_ns_diffusion_40s_sto_midV_256.h5'
with h5py.File(filename, 'r') as file:
    sol_t = torch.tensor(file['t'][()], device='cuda')
    sol = torch.tensor(file['sol'][()], device='cuda')
    diffusion = torch.tensor(file['diffusion'][()], device='cuda')

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
Ntest = 1000

train_diffusion_256 = diffusion_reshaped_train[:Ntrain, :, :]
train_vorticity_256= sol_reshaped_train[:Ntrain, :, :]

test_diffusion_256 = diffusion_reshaped_test[:Ntest, :, :]
test_vorticity_256 = sol_reshaped_test[:Ntest, :, :]

train_diffusion_64 = train_diffusion_256[:, ::4, ::4]
train_vorticity_64 = train_vorticity_256[:, ::4, ::4]

test_diffusion_64 = test_diffusion_256[:, ::4, ::4]
test_vorticity_64 = test_vorticity_256[:, ::4, ::4]

test_diffusion_128 = test_diffusion_256[:, ::2, ::2]
test_vorticity_128 = test_vorticity_256[:, ::2, ::2]

test_diffusion_64_sparse = test_diffusion_64[:, ::4, ::4]
test_diffusion_128_sparse = test_diffusion_128[:, ::8, ::8]
test_diffusion_256_sparse = test_diffusion_256[:, ::16, ::16]

filename = 'C:\\UWMadisonResearch\\Conditional_Score_FNO\\Data_Generation\\train_diffusion.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('train_diffusion_64', data=train_diffusion_64.cpu().numpy())
    file.create_dataset('train_vorticity_64', data=train_vorticity_64.cpu().numpy())

filename = 'C:\\UWMadisonResearch\\Conditional_Score_FNO\\Data_Generation\\test_diffusion.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('test_diffusion_64', data=test_diffusion_64.cpu().numpy())
    file.create_dataset('test_vorticity_64', data=test_vorticity_64.cpu().numpy())
    file.create_dataset('test_diffusion_128', data=test_diffusion_128.cpu().numpy())
    file.create_dataset('test_vorticity_128', data=test_vorticity_128.cpu().numpy())
    file.create_dataset('test_diffusion_256', data=test_diffusion_256.cpu().numpy())
    file.create_dataset('test_vorticity_256', data=test_vorticity_256.cpu().numpy())

