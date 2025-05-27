
# DQLoRA: Adapter-Guided Distillation for Lightweight ASR

- With Python 3 + A100 GPU, deployed on Colab

# 1. Install Dependencies and Import Libraries

```bash
!pip install -q transformers datasets torchaudio peft accelerate bitsandbytes
import torch
import torchaudio
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    WhisperProcessor,
    WhisperModel,
    AdamW
)
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
```


# 2. Load FLEURS (Clean) + DNS (Noise)

```
!pip install -U datasets
!pip install fsspec==2023.9.2

from datasets import load_dataset
fleurs = load_dataset("google/fleurs", name="en_us", split="train", streaming=True)

!rm -rf ~/.cache/huggingface/datasets
```
<br>

```
import os
import shutil
from datasets import load_dataset, load_from_disk, Audio

# Step 0: Clear known HuggingFace environment caches
os.environ.pop("HF_DATASETS_CACHE", None)
os.environ.pop("HF_HOME", None)

# Step 1: Clear manually any existing Drive cache (for safety)
drive_cache = "/content/drive/MyDrive/.cache/huggingface"
if os.path.exists(drive_cache):
    shutil.rmtree(drive_cache)
    print("Removed stale cache from Google Drive")

# Step 2: Define paths
hf_cache_dir = "/content/hf_cache"
fleurs_temp_path = "/content/fleurs_subset"
fleurs_drive_path = "/content/drive/MyDrive/data/fleurs_subset"

# Step 3: Download FLEURS only if not already processed
if not os.path.exists(fleurs_drive_path):
    print("Downloading fresh FLEURS dataset...")

    # Force download by setting download_mode="force_redownload"
    fleurs = load_dataset("google/fleurs", "en_us", split="train", cache_dir=hf_cache_dir, download_mode="force_redownload")
    fleurs = fleurs.cast_column("audio", Audio(sampling_rate=16000))
    fleurs_subset = fleurs.select(range(100))  # Use subset

    # Save locally then copy to Drive
    fleurs_subset.save_to_disk(fleurs_temp_path)
    shutil.copytree(fleurs_temp_path, fleurs_drive_path)
    print(f"FLEURS saved to: {fleurs_drive_path}")
else:
    print(f"FLEURS already exists at: {fleurs_drive_path}")

# Step 4: Load and check
fleurs_loaded = load_from_disk(fleurs_drive_path)
print(f"FLEURS loaded successfully. Samples: {len(fleurs_loaded)}")
```


# 3. Load Student Model (Wav2Vec2 + QLoRA)

```
for name, module in student_model.named_modules():
    if "attention" in name:
        print(name)
```

```
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import torch

# QLoRA quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load quantized model
student_model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    device_map="auto",
    quantization_config=bnb_config
)

# Build all attention projection modules for 12 layers
target_modules = []
for i in range(12):
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        target_modules.append(f"wav2vec2.encoder.layers.{i}.attention.{proj}")

# LoRA config
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=target_modules
)

# Inject adapters
student_model = get_peft_model(student_model, lora_cfg)
student_model.train()

# Load processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
```



# 4. Load Teacher (Whisper Encoder)

```
teacher_model = WhisperModel.from_pretrained("openai/whisper-small").to("cuda").eval()
teacher_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
```


# 5. Data Preprocessing: Clean + Noisy + Labels

```
def add_noise(clean, noise, snr_db=5):
    clean = clean[:len(noise)]
    snr = 10 ** (snr_db / 10)
    signal_power = np.mean(clean ** 2)
    noise_power = signal_power / snr
    noise = noise * (noise_power / np.mean(noise ** 2))**0.5
    return clean + noise

def preprocess(batch):
    speech = batch["audio"]["array"]
    noise = dns[0]["audio"]["array"]
    speech_noisy = add_noise(np.array(speech), np.array(noise))

    inputs = processor(speech_noisy, sampling_rate=16000, return_tensors="pt", padding=True)
    labels = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True).input_values[0]
    inputs["labels"] = labels
    return inputs

dataloader = DataLoader(fleurs, batch_size=1, collate_fn=lambda x: preprocess(x[0]))
```


# 6. Training Loop (Distillation + CTC)

```
optimizer = AdamW(student_model.parameters(), lr=1e-4)
lambda_distill = 0.7

for step, batch in enumerate(tqdm(dataloader)):
    batch = {k: v.to("cuda") for k, v in batch.items()}

    # Teacher: Get logits
    with torch.no_grad():
        teacher_outputs = teacher_model.encoder(batch["input_values"])
        teacher_embeds = teacher_outputs.last_hidden_state

    # Student: Forward pass
    student_outputs = student_model(**batch)
    student_logits = student_outputs.logits

    # Loss: CTC + Distillation
    ctc_loss = F.ctc_loss(student_logits.log_softmax(dim=-1), batch["labels"], torch.full((1,), student_logits.shape[1], dtype=torch.long).to("cuda"), torch.full((1,), batch["labels"].shape[1], dtype=torch.long).to("cuda"), blank=processor.tokenizer.pad_token_id, reduction='mean')

    with torch.no_grad():
        whisper_logits = teacher_model(batch["input_values"]).logits

    distill_loss = F.kl_div(student_logits.log_softmax(dim=-1), whisper_logits.softmax(dim=-1), reduction="batchmean")
    total_loss = ctc_loss + lambda_distill * distill_loss

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step > 2:
        break

print("QLoRA + Whisper distillation demo complete.")
```

<br><br>

## Citation

```bib
@article{yiruyang2025,
  title={A Lightweight Domain-Aware Denoising ASR via Adapter-guided Distillation},
  author={Yiru Yang, xxxx},
  journal={arXiv preprint arXiv:2506.xxxxx},
  year={2025}
}
```

<br><br><br><br>
