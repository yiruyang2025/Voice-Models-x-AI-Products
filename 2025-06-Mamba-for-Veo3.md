
1. Install Dependencies
```
import os
import subprocess

# Clone the repo if not already present
if not os.path.exists("Text2Video-Zero"):
    !git clone https://github.com/Picsart-AI-Research/Text2Video-Zero.git
%cd Text2Video-Zero

# Check if requirements are installed
try:
    import transformers
    import diffusers
    import einops
    print("Dependencies already installed.")
except ImportError:
    print("Installing dependencies...")
    !pip install -r requirements.txt

# Install Mamba if not already installed
try:
    import mamba_ssm
    print("mamba_ssm already installed.")
except ImportError:
    print("Installing mamba_ssm...")
    !pip install git+https://github.com/state-spaces/mamba.git
```


2. Define Mamba-1 lightweight memory module
```
# Define the Mamba block wrapper
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaInjectedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(d_model=dim)

    def forward(self, x):
        return self.mamba(x)
```

3. Inject Mamba into Veo3-like U-Net model
```
# Example modified U-Net block using Mamba
class ModifiedUNetBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mamba = MambaInjectedBlock(in_dim)

    def forward(self, x):
        # Expected input shape: [batch, sequence_length, feature_dim]
        return self.mamba(x)

# Sanity check
x = torch.randn(2, 128, 64)  # batch=2, 128 frames, 64-dim latent features
model = ModifiedUNetBlock(in_dim=64)
out = model(x)
print("Mamba output shape:", out.shape)
```

4. Mock Mamba + Latent Video Consistency Experiment
```
# Simulate latent video sequence across 16 frames per batch
latent_video = torch.randn(4, 16, 128)  # batch=4, 16 frames, 128-dimensional latent vector

# Apply Mamba module over the temporal axis
temporal_mamba = MambaInjectedBlock(dim=128)
refined_video = temporal_mamba(latent_video)

# Compare pre/post transformation statistics
print("Original latent mean:", latent_video.mean().item())
print("Refined latent mean:", refined_video.mean().item())
```

5. Evaluate temporal face consistency
```
# Simulate cosine similarity measurement between consecutive frames
def cosine_similarity(a, b):
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_norm * b_norm).sum(dim=-1).mean().item()

original_score = cosine_similarity(latent_video[:, :-1, :], latent_video[:, 1:, :])
refined_score = cosine_similarity(refined_video[:, :-1, :], refined_video[:, 1:, :])

print("Original temporal consistency score:", original_score)
print("Refined temporal consistency score:", refined_score)
```


6. Report result table
```
import pandas as pd

data = {
    "Model Variant": ["Base Model", "Base + Mamba-1"],
    "Mean Latent": [latent_video.mean().item(), refined_video.mean().item()],
    "Temporal Consistency Score": [original_score, refined_score]
}

df = pd.DataFrame(data)
print(df)
```



```
```
