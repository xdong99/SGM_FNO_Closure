import sys
sys.path.append('C:\\UWMadisonResearch\\Conditional_Score_FNO')
import torch
import math
from tqdm.notebook import tqdm
from Data_Generation.random_forcing import get_twod_bj, get_twod_dW

# a: domain where we are solving
# w0: initial vorticity
# f: deterministic forcing term
# visc: viscosity (1/Re)
# T: final time
# delta_t: internal time-step for solve (descrease if blow-up)
# record_steps: number of in-time snapshots to record

def navier_stokes_2d(a, w0, f, visc, T, delta_t, record_steps, stochastic_forcing=None):
    # Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1]

    # Maximum frequency
    k_max1 = math.floor(N1 / 2.0)
    k_max2 = math.floor(N1 / 2.0)

    # Number of steps to final time
    steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1, 2])
    w_h = torch.stack([w_h.real, w_h.imag], dim=-1)

    # Forcing to Fourier space
    if f is not None:
        f_h = torch.fft.fftn(f, dim=[-2, -1])
        f_h = torch.stack([f_h.real, f_h.imag], dim=-1)
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
    else:
        f_h = torch.zeros_like(w_h)


    # If stochastic forcing
    if stochastic_forcing is not None:
        # initialise noise
        bj = get_twod_bj(delta_t, [N1, N2], a, stochastic_forcing['alpha'], w_h.device)


    # Record solution every this number of steps
    record_time = math.floor(steps / record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max2, step=1, device=w0.device),
                     torch.arange(start=-k_max2, end=0, step=1, device=w0.device)), 0).repeat(N1, 1).transpose(0, 1)
    # Wavenumbers in x-direction
    k_x = torch.cat((torch.arange(start=0, end=k_max1, step=1, device=w0.device),
                     torch.arange(start=-k_max1, end=0, step=1, device=w0.device)), 0).repeat(N2, 1)
    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 / a[0] ** 2 + k_y ** 2 / a[1] ** 2)

    lap[0, 0] = 1.0

    # Dealiasing mask
    dealias = torch.unsqueeze(
        torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max2, torch.abs(k_x) <= (2.0 / 3.0) * k_max1).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps+1, device=w0.device)
    sol_u = torch.zeros(*w0.size(), record_steps+1, device=w0.device)
    sol_v = torch.zeros(*w0.size(), record_steps+1, device=w0.device)
    forcing = torch.zeros(*w0.size(), record_steps+1, device=w0.device)
    diffusion = torch.zeros(*w0.size(), record_steps+1, device=w0.device)
    sol_t = torch.zeros(record_steps+1, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in tqdm(range(steps+1)):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h = psi_h.to(dtype=torch.float64)
        psi_h[..., 0] = psi_h[..., 0] / lap
        psi_h[..., 1] = psi_h[..., 1] / lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[..., 0].clone()
        q[..., 0] = -2 * math.pi * k_y * q[..., 1]
        q[..., 1] = 2 * math.pi * k_y * temp
        q = torch.fft.ifftn(torch.view_as_complex(q / a[1]), dim=[1, 2], s=(N1, N2)).real

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[..., 0].clone()
        v[..., 0] = 2 * math.pi * k_x * v[..., 1]
        v[..., 1] = -2 * math.pi * k_x * temp
        v = torch.fft.ifftn(torch.view_as_complex(v / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[..., 0].clone()
        w_x[..., 0] = -2 * math.pi * k_x * w_x[..., 1]
        w_x[..., 1] = 2 * math.pi * k_x * temp
        w_x = torch.fft.ifftn(torch.view_as_complex(w_x / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[..., 0].clone()
        w_y[..., 0] = -2 * math.pi * k_y * w_y[..., 1]
        w_y[..., 1] = 2 * math.pi * k_y * temp
        w_y = torch.fft.ifftn(torch.view_as_complex(w_y / a[1]), dim=[1, 2], s=(N1, N2)).real

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.fftn(q * w_x + v * w_y, dim=[1, 2])
        F_h = torch.stack([F_h.real, F_h.imag], dim=-1)

        # Dealias
        F_h[..., 0] = dealias * F_h[..., 0]
        F_h[..., 1] = dealias * F_h[..., 1]

        # Cranck-Nicholson update
        w = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1, 2], s=(N1, N2)).real

        if stochastic_forcing:
            dW, dW2 = get_twod_dW(bj, stochastic_forcing['kappa'], w_h.shape[0], w_h.device)
            gudWh = stochastic_forcing['sigma'] * torch.fft.fft2(dW, dim=[-2, -1])
            gudWh = torch.stack([gudWh.real, gudWh.imag], dim=-1)
        else:
            gudWh = torch.zeros_like(f_h)

        diffusion_h = w_h.clone()
        w_h_mse = torch.mean(w_h[..., 0] ** 2)
        gudWh_mse = torch.mean(gudWh[..., 0] ** 2)
        print(w_h_mse, gudWh_mse)

        diffusion_h[..., 0] = -visc * lap * w_h[..., 0] + 2 * gudWh[..., 0]
        diffusion_h[..., 1] = -visc * lap * w_h[..., 1] + 2 * gudWh[..., 1]

        # Update real time (used only for recording)
        if j == 0:
            diffusion_term = torch.fft.ifftn(torch.view_as_complex(diffusion_h), dim=[1, 2], s=(N1, N2)).real
            sol[..., 0] = w
            sol_u[..., 0] = q
            sol_v[..., 0] = v
            if stochastic_forcing:
                forcing[..., 0] = stochastic_forcing['sigma'] * dW
            diffusion[..., 0] = diffusion_term
            sol_t[0] = 0

            c += 1

        if j !=0 and (j) % record_time == 0:
            # Solution in physical space
            diffusion_term = torch.fft.ifftn(torch.view_as_complex(diffusion_h), dim=[1, 2], s=(N1, N2)).real

            if stochastic_forcing:
                forcing[..., c] = stochastic_forcing['sigma'] * dW
            # Record solution and time
            sol[..., c] = w
            sol_u[..., c] = q
            sol_v[..., c] = v
            sol_t[c] = t
            diffusion[..., c] = diffusion_term

            c += 1

        t += delta_t

        w_h[..., 0] = ((w_h[..., 0] -delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + 0.5 * delta_t * diffusion_h[..., 0])
                      / (1.0 + 0.5 * delta_t * visc * lap))
        w_h[..., 1] = ((w_h[..., 1] -delta_t * F_h[..., 1] + delta_t * f_h[..., 1] + 0.5 * delta_t * diffusion_h[..., 1])
                      / (1.0 + 0.5 * delta_t * visc * lap))


    if stochastic_forcing:
        return sol, sol_u, sol_v, sol_t, forcing, diffusion

    return sol, sol_u, sol_v, sol_t, diffusion
