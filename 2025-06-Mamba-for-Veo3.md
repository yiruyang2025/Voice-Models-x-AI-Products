
1. Install Dependencies
```
!git clone https://github.com/Picsart-AI-Research/Text2Video-Zero.git
%cd Text2Video-Zero
!pip install -r requirements.txt
!pip install mamba-ssm
```


2. Define Mamba-1 lightweight memory module
```
# Install lightweight Mamba (pure Python variant)
!pip install git+https://github.com/state-spaces/mamba.git
```

3. Inject Mamba into Veo3-like U-Net model
```
import torch
import torch.nn as nn
from mamba_ssm import Mamba

# Create a simplified Mamba block wrapper
class MambaInjectedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(d_model=dim)

    def forward(self, x):
        return self.mamba(x)

# Example: Replace a U-Net attention block with Mamba
class ModifiedUNetBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mamba = MambaInjectedBlock(in_dim)

    def forward(self, x):
        # Assumes input shape: [batch, sequence, features]
        return self.mamba(x)

# Sanity check
x = torch.randn(2, 128, 64)  # e.g. batch=2, sequence_len=128, dim=64
model = ModifiedUNetBlock(in_dim=64)
out = model(x)
print("Mamba output shape:", out.shape)
```

4. Mock Mamba + Latent Video Consistency Experiment
```
# Simulate latent video features over frames
latent_video = torch.randn(4, 16, 128)  # batch=4, 16 frames, 128-dim latent vector

# Apply Mamba temporal smoothing (mock)
temporal_mamba = MambaInjectedBlock(dim=128)
refined_video = temporal_mamba(latent_video)

# Compare pre/post
print("Before:", latent_video.mean().item(), "After:", refined_video.mean().item())
```

5. Evaluate temporal face consistency
```

```


6. Report result table
```
```



```
```
