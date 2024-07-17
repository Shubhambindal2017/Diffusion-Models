import torch
import numpy as np

class DDPMVanilla:
    def __init__(self, timesteps, beta1=1e-4, beta2=0.02, device='cuda'):
        # construct DDPM noise schedule
        # Basically these s1, s2, s3 - samples from DDPM algorithm paper
        b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
        a_t = 1 - b_t
        ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
        ab_t[0] = 1

        self.ab_t = ab_t
        self.a_t = a_t
        self.b_t = b_t
        self.device = device
        self.timesteps = timesteps

    def denoise_add_noise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        ## Imortant to note that pred_noise is subtract from x and later some noise is also added
        ## other things are mostly from DDPM paper
        return mean + noise

    # sample using standard algorithm
    @torch.no_grad()
    def sample_ddpm(self, model, height, n_sample, save_rate=20):
        # x_T ~ N(0, 1), sample initial noise
        timesteps = self.timesteps
        samples = torch.randn(n_sample, 3, height, height).to(self.device)  

        # array to keep track of generated steps for plotting
        intermediate = [] 
        for i in range(timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / timesteps])[:, None, None, None].to(self.device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps = model(samples, t)    # predict noise e_(x_t,t)
            samples = self.denoise_add_noise(samples, i, eps, z)
            if i % save_rate ==0 or i==timesteps or i<8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate