import numpy as np
import torch
from torch.distributions import Beta
import math

# Define the logit-normal distribution, used for plotting the density
def logit_normal_density(t, m=0.0, s=1.0):
    """Theoretical density function"""
    t = np.clip(t, 1e-10, 1-1e-10)
    logit_t = np.log(t/(1-t))
    density = (1/(s*np.sqrt(2*np.pi))) * (1/(t*(1-t))) * np.exp(-(logit_t - m)**2 / (2*s**2))
    return density

def f_mode(u, s):
    """
    Implements the mode sampling function from equation (20) in the paper.
    Args:
        u: Input tensor in range [0,1]
        s: Scale parameter controlling sampling behavior
    Returns:
        Transformed values according to the mode function
    """
    return 1 - u - s * (torch.cos(torch.pi/2 * u)**2 - 1 + u)


def sample_logit_normal(batch_size, m=0.0, s=1.0, device='cuda'):
    """
    Generate samples from a Logit-Normal distribution.
    
    Args:
        batch_size: Number of samples to generate
        m: Location parameter (default 0.0)
            - negative m biases towards 0
            - positive m biases towards 1
        s: Scale parameter (default 1.0) controls spread of distribution
        device: Device to place tensors on
    
    Returns:
        t: Tensor of shape (batch_size, 1, 1) with values in (0,1)
    """
    # Sample u from normal distribution N(m, s)
    u = torch.normal(mean=m, std=s, size=(batch_size, 1, 1), device=device)
    
    # Transform through logistic function: sigmoid(u) = 1 / (1 + exp(-u))
    t = torch.sigmoid(u)
    
    # # Clamp to prevent numerical instability
    # t = t.clamp(1e-7, 1 - 1e-7)
    
    return t

def sample_mode(batch_size, s=1.29, device='cuda'):
    """
    Samples timesteps using the mode sampling distribution.
    Args:
        batch_size: Number of samples to generate
        s: Scale parameter in range [-1, 2/π²]
        device: Device to place tensors on
    Returns:
        Tensor of shape (batch_size, 1, 1) containing sampled timesteps
    """
    # Generate uniform samples as starting point
    u = torch.rand(batch_size, 1, 1, device=device)
    
    # Apply the mode sampling function
    t = f_mode(u, s)
    
    # Ensure outputs are in [0,1] range
    t = torch.clamp(t, 0, 1)
    
    return t

def sample_cosmap(batch_size, device='cuda'):
    """
    Samples timesteps using the cosine schedule sampling distribution.
    Args:
        batch_size: Number of samples to generate
        device: Device to place tensors on
    Returns:
        Tensor of shape (batch_size, 1, 1) containing sampled timesteps
    """
    # Generate uniform samples as starting point
    u = torch.rand(batch_size, 1, 1, device=device)
    
    # Apply the cosmap sampling function from equation (21)
    # t = 1 - 1/(tan(πu/2) + 1)
    t = 1 - 1/(torch.tan(torch.pi/2 * u) + 1)
    
    # Ensure outputs are in [0,1] range and handle numerical stability
    t = torch.clamp(t, 0, 1)
    
    return t

def sample_beta(batch_size, s=0.999, alpha=1.0, beta=1.5, device='cuda'):
    """
    Samples timesteps using a shifted beta distribution with cutoff.
    Args:
        batch_size: Number of samples to generate
        s: Cutoff threshold (default 0.999 as used in paper)
        alpha: Beta distribution alpha parameter (default 1.0 as used in paper)
        beta: Beta distribution beta parameter (default 1.5 as used in paper)
        device: Device to place tensors on
    Returns:
        Tensor of shape (batch_size, 1, 1) containing sampled timesteps
    """
    # Create beta distribution
    beta_dist = Beta(torch.tensor([alpha], device=device), 
                    torch.tensor([beta], device=device))
    
    # Sample from beta distribution
    raw_samples = beta_dist.sample((batch_size, 1, 1))
    
    # Scale samples by s to get timesteps in [0, s]
    t = s * raw_samples
    
    return t

def sample_discrete_pow(batch_size, denoise_timesteps, device="cuda"):
   log2_sections = int(math.log2(denoise_timesteps)) + 1  # 8 for 128
   
   # 反转并重复
   dt_base = np.repeat(log2_sections - 1 - np.arange(log2_sections), 
                      batch_size // log2_sections)
   
   # 填充0以匹配batch_size
   dt_base = np.concatenate([dt_base, np.zeros(batch_size - dt_base.shape[0])])
   
   # 计算对应的sections
   dt_sections = 2 ** dt_base
   
   # 对每个样本采样
   t = np.random.randint(0, dt_sections, size=batch_size).astype(np.float32)
   t = t / dt_sections

   t = torch.tensor(t, device=device).float()

   return t
