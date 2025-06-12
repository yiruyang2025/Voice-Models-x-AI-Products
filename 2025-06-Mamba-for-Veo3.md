
1. Install Dependencies
```
!git clone https://github.com/Picsart-AI-Research/Text2Video-Zero.git
%cd Text2Video-Zero
!pip install -r requirements.txt
!pip install mamba-ssm
```


2. Define Mamba-1 lightweight memory module
```
import torch.nn as nn
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, dim=512, d_state=1, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=dim, d_state=d_state, expand=expand)

    def forward(self, x):
        return self.mamba(x)
```

3. Inject Mamba into Veo3-like U-Net model
```
from model.unet import UNet3DConditionModel

def inject_mamba(model, dim=512):
    mamba_layer = MambaBlock(dim)
    for name, module in model.named_modules():
        if 'cross_frame_attn' in name:
            print(f"Replacing {name} with MambaBlock")
            setattr(model, name, mamba_layer)
```

4. Generate video from text
```
from inference.pipeline import text2video_infer

text_prompt = "A man wearing a red jacket walking through a snowy forest"
video = text2video_infer(prompt=text_prompt)
video.save("veo3_mamba1_output.mp4")
```

5. Evaluate temporal face consistency
```
from insightface.app import FaceAnalysis
import numpy as np

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)
frames = extract_frames("veo3_mamba1_output.mp4")

embs = [app.get(frame)[0].embedding for frame in frames if app.get(frame)]
cos_sim = np.mean([np.dot(embs[i], embs[i+1]) for i in range(len(embs)-1)])

print("Temporal Identity Consistency Score:", round(cos_sim, 4))
```


6. Report result table
```
```


```
```
