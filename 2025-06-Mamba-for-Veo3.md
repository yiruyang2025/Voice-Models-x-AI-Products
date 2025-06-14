
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
# Install Mamba-SSM (lightweight, skip CUDA deps)
try:
    import mamba_ssm
    print("mamba_ssm is already installed.")
except ImportError:
    print("Installing mamba_ssm without heavy CUDA dependencies...")
    # install the build tool once
    !pip install -q ninja
    # install Mamba-SSM without reloading CUDA wheels
    !pip install -q git+https://github.com/state-spaces/mamba.git#egg=mamba_ssm --no-deps
```

```
# Cell 2: Import or define Mamba, then other libs

try:
    from mamba_ssm import Mamba
    print("Using real mamba_ssm.Mamba")
except ImportError:
    print("mamba_ssm not found, using placeholder Mamba")
    class Mamba:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

import torch
import torch.nn as nn
import pandas as pd
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

```
@article{yiruyang2025,
  title={Mamba For Veo 3},
  author={Yiru Yang},
  journal={arXiv preprint arXiv:2505.xxxxx},
  year={2025}
}
```
