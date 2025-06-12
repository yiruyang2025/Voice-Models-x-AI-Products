
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
```

```
# Install Mamba if not already installed
try:
    import mamba_ssm
    print("mamba_ssm already installed.")
except ImportError:
    print("Installing mamba_ssm...")
    !pip install git+https://github.com/state-spaces/mamba.git
```

```
# Cell 2: Imports
import torch
import torch.nn as nn
import pandas as pd
from mamba_ssm import Mamba
```

```
# Cell 3: Define Mamba-1 module
class MambaInjectedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(d_model=dim)
    def forward(self, x):
        return self.mamba(x)
```


```
# Cell 4: Sanity check
class ModifiedUNetBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.block = MambaInjectedBlock(in_dim)
    def forward(self, x):
        return self.block(x)

x = torch.randn(2, 128, 64)
out = ModifiedUNetBlock(64)(x)
print("output shape", out.shape)
```

```
# Cell 5: Mock experiment setup
latent_video = torch.randn(4, 16, 128)
refined_video = MambaInjectedBlock(128)(latent_video)
print("mean latent", latent_video.mean().item(), refined_video.mean().item())
```

```
# Cell 6: Temporal consistency
def cosine_similarity(a, b):
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return float((a * b).sum(dim=-1).mean())

orig_score = cosine_similarity(latent_video[:, :-1], latent_video[:, 1:])
ref_score = cosine_similarity(refined_video[:, :-1], refined_video[:, 1:])
print("temporal scores", orig_score, ref_score)
```


```
# Cell 7: Results table
df = pd.DataFrame({
    "Model": ["Base", "Base+Mamba1"],
    "MeanLatent": [latent_video.mean().item(), refined_video.mean().item()],
    "TemporalScore": [orig_score, ref_score]
})
print(df)
```
