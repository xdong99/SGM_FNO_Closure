import sys
sys.path.append('C:\\UWMadisonResearch\\Conditional_Score_FNO')

import torch
import numpy as np
import math
import h5py
from timeit import default_timer

from Data_Generation.generator_sns import navier_stokes_2d
from Data_Generation.random_forcing import GaussianRF

filename = "./Data_Generation/"
device = torch.device('cuda')

# Viscosity parameter
nu = 1e-3

# Spatial Resolution
s = 256
sub = 1

# Temporal Resolution
T = 40
delta_t = 1e-3

# Number of solutions to generate
N = 10

# Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s + 1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

# Stochastic forcing function: sigma*dW/dt
stochastic_forcing = {'alpha': 0.005, 'kappa': 10, 'sigma': 0.00005}

# Number of snapshots from solution
record_steps = 4000

# Solve equations in batches (order of magnitude speed-up)
# Batch size
bsize = 10

c = 0
t0 = default_timer()

for j in range(N // bsize):
    w0 = GRF.sample(bsize)

    sol, sol_u, sol_v, sol_t, forcing, diffusion = navier_stokes_2d([1, 1], w0, f, nu, T, delta_t, record_steps, stochastic_forcing = stochastic_forcing)

    c += bsize
    t1 = default_timer()
    print(j, c, t1 - t0)

sol_t_np = sol_t.cpu().numpy()
sol_np = sol.cpu().numpy()
sol_u_np = sol_u.cpu().numpy()
sol_v_np = sol_v.cpu().numpy()
forcing_np = forcing.cpu().numpy()
diffusion_np = diffusion.cpu().numpy()
nu_np = np.array(nu)

# Specify the filename for the HDF5 file
filename = '2d_ns_diffusion_40s_sto_midV_256.h5'

# Create a new HDF5 file
with h5py.File(filename, 'w') as file:
    # Create datasets within the file
    file.create_dataset('t', data=sol_t_np)
    file.create_dataset('sol', data=sol_np)
    file.create_dataset('sol_u', data=sol_u_np)
    file.create_dataset('sol_v', data=sol_v_np)
    file.create_dataset('forcing', data=forcing_np)
    file.create_dataset('diffusion', data=diffusion_np)
    file.create_dataset('nu', data=nu_np)

print(f'Data saved to {filename}')