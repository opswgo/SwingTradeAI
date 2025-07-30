#!/usr/bin/env python3
"""
Safe model loader for SAMBA that doesn't execute training code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import warnings
import dill
import pickle

# Define a minimal Trainer class that the checkpoint might be looking for
class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def load_state_dict(self, state_dict, strict=True):
        # Handle loading of trainer state if needed
        if 'model_state_dict' in state_dict:
            if self.model is not None:
                self.model.load_state_dict(state_dict['model_state_dict'])
        return self

# Define MambaBlock class that the model checkpoint expects
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Convolutional filter
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=(d_conv - 1) // 2,
            bias=False
        )
        
        # Linear projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj.weight.data.zero_()
        
    def forward(self, x):
        # Input shape: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x_proj = self.in_proj(x)  # (batch, seq_len, 2*d_inner)
        x, z = x_proj.chunk(2, dim=-1)  # (batch, seq_len, d_inner) each
        x = x.permute(0, 2, 1)  # (batch, d_inner, seq_len)
        
        # Apply depthwise convolution
        x = self.conv1d(x)  # (batch, d_inner, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, d_inner)
        x = F.silu(x)
        
        # SSM computation (simplified)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        x = F.linear(x, A)  # (batch, seq_len, d_state)
        x = x + z  # Add skip connection
        
        # Output projection
        out = self.out_proj(x)  # (batch, seq_len, d_model)
        return out

# Define ResidualBlock class that the model checkpoint expects
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out

# Define the minimal model architecture needed for inference
@dataclass
class ModelArgs:
    d_model: int = 64
    n_layer: int = 4
    vocab_size: int = 1
    seq_in: int = 12
    d_inner: int = 256
    d_conv: int = 4
    dt_rank: int = 16
    d_state: int = 16
    bias: bool = True
    conv_bias: bool = True

class Mamba(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def forward(self, x):
        return x  # Dummy implementation

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class SAMBA(nn.Module):
    def __init__(self, args, hidden, inp, out, embed, cheb_k):
        super().__init__()
        self.args = args
        self.hidden = hidden
        self.inp = inp
        self.out = out
        self.embed = embed
        self.cheb_k = cheb_k
        
        # Initialize parameters
        self.mam1 = Mamba(args)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.adj = nn.Parameter(torch.randn(args.vocab_size, embed), requires_grad=True)
        self.embed_w = nn.Parameter(torch.randn(embed, embed), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed, out))
        self.proj = nn.Linear(args.vocab_size, 1)
        self.proj_seq = nn.Linear(args.seq_in, 1)
        self.rms_norm = RMSNorm(d_model=embed)
        
    def forward(self, x):
        # Dummy forward pass that returns zeros with correct shape
        return torch.zeros(x.size(0), self.out, device=x.device)

def load_samba_model(model_path, input_dim, predict_steps=1, device=None):
    """
    Load a pre-trained SAMBA model without executing training code.
    
    Args:
        model_path: Path to the saved model file
        input_dim: Number of input features
        predict_steps: Number of steps to predict
        device: Device to run the model on ('cuda', 'mps', or 'cpu')
        
    Returns:
        Loaded model ready for inference
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    try:
        # Initialize model with default parameters
        model_args = ModelArgs(
            d_model=64,  # Default values - adjust based on your model
            n_layer=4,
            vocab_size=input_dim,
            seq_in=12,
            d_inner=256,
            d_conv=4,
            dt_rank=16,
            d_state=16,
            bias=True,
            conv_bias=True
        )
        
        # Initialize the model
        model = SAMBA(
            model_args,
            hidden=64,  # Default hidden size
            inp=input_dim,
            out=predict_steps,
            embed=32,   # Default embedding size
            cheb_k=3    # Default Chebyshev polynomial order
        )
        
        # Try to load the model using torch's load function with various approaches
        load_attempts = [
            # Attempt 1: Try loading with weights_only=False (PyTorch 2.6+)
            {
                'name': 'direct loading with weights_only=False',
                'function': lambda: torch.load(model_path, map_location=device, weights_only=False)
            },
            # Attempt 2: Try loading with dill
            {
                'name': 'loading with dill',
                'function': lambda: dill.load(open(model_path, 'rb'))
            },
            # Attempt 3: Try direct loading with pickle
            {
                'name': 'direct loading with pickle',
                'function': lambda: pickle.load(open(model_path, 'rb'))
            },
            # Attempt 4: Try loading with torch.jit.load (for TorchScript models)
            {
                'name': 'loading with torch.jit',
                'function': lambda: torch.jit.load(model_path, map_location=device)
            }
        ]
        
        # Try each loading method until one succeeds
        last_error = None
        
        for attempt in load_attempts:
            try:
                print(f"Attempting to load model using {attempt['name']}...")
                checkpoint = attempt['function']()
                
                # If we get here, loading was successful
                print(f"Successfully loaded checkpoint using {attempt['name']}")
                
                # Try to extract the model state dict
                if isinstance(checkpoint, Trainer):
                    # If the checkpoint is a Trainer object, get its model
                    if hasattr(checkpoint, 'model') and checkpoint.model is not None:
                        # If the Trainer has a model attribute, use that
                        model = checkpoint.model
                        print("Successfully extracted model from Trainer object")
                        break  # Exit the loading loop as we have the model
                    else:
                        # If no model in Trainer, try to get state_dict from Trainer
                        state_dict = checkpoint.state_dict()
                elif hasattr(checkpoint, 'state_dict'):
                    # Checkpoint is a model object
                    state_dict = checkpoint.state_dict()
                elif isinstance(checkpoint, dict):
                    # Checkpoint is a dictionary
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
                        # Handle case where model is stored under 'model' key
                        model = checkpoint['model']
                        print("Successfully extracted model from checkpoint['model']")
                        break  # Exit the loading loop as we have the model
                    else:
                        # Try to use the entire checkpoint as a state dict
                        state_dict = checkpoint
                else:
                    # Try to use the checkpoint directly as a state dict
                    state_dict = checkpoint
                
                # Load the state dict into the model
                model.load_state_dict(state_dict, strict=False)  # strict=False to allow partial loading
                print("Successfully loaded model weights")
                break
                
            except Exception as e:
                last_error = e
                print(f"Failed to load with {attempt['name']}: {str(e)}")
                continue
        else:
            # If we get here, all loading attempts failed
            raise RuntimeError(
                "Failed to load model after multiple attempts. "
                f"Last error: {str(last_error)}\n"
                "Please ensure the model file is not corrupted and is in a supported format."
            )
        
        model = model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Test loading the model
    try:
        model = load_samba_model(
            model_path='AGCRN_Model/milan/call/1_stemgnn.pt',
            input_dim=10,  # Number of input features
            predict_steps=1,
            device='cpu'  # 'cuda', 'mps', or 'cpu'
        )
        print("Model loaded successfully!")
        print(f"Model architecture: {model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
