
# DQLoRA: Adapter-Guided Distillation for Lightweight ASR

- With Python 3 + A100 GPU, deployed on Colab

# 1. Install Dependencies and Import Libraries

```bash
!pip install -q transformers datasets torchaudio peft accelerate bitsandbytes
import torch
import torchaudio
import numpy as np
from datasets import load_dataset, Audio
from torch.optim import AdamW

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    WhisperProcessor,
    WhisperModel
)

from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType
)

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
from transformers import WhisperModel, WhisperProcessor

# Load Whisper encoder as teacher (frozen)
teacher_model = WhisperModel.from_pretrained("openai/whisper-small").to("cuda").eval()
teacher_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
```


# 5. Data Preprocessing: Clean + Noisy + Labels

```
import numpy as np
from torch.utils.data import DataLoader

# Add noise to clean speech with specified SNR (dB)
def add_noise(clean, noise, snr_db=5):
    clean = clean[:len(noise)]  # Align length
    snr = 10 ** (snr_db / 10)
    signal_power = np.mean(clean ** 2)
    noise_power = signal_power / snr
    noise = noise * (noise_power / np.mean(noise ** 2)) ** 0.5
    return clean + noise

# Preprocess a single example: clean + noisy + labels
def preprocess(batch):
    speech = batch["audio"]["array"]
    noise = dns[0]["audio"]["array"]  # First DNS sample
    speech_noisy = add_noise(np.array(speech), np.array(noise))

    inputs = processor(
        speech_noisy,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    labels = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values[0]

    inputs["labels"] = labels
    return inputs

# Wrap with DataLoader
dataloader = DataLoader(
    fleurs_loaded,  # or fleurs_subset if used
    batch_size=1,
    collate_fn=lambda x: preprocess(x[0])
)
```


# 6. Training Loop (Distillation + CTC)

```
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WhisperProcessor, WhisperModel

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
student_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda").half()

teacher_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
teacher_model = WhisperModel.from_pretrained("openai/whisper-base").to("cuda").eval()
```

```
print(f"[Step {step}] input_len={input_lengths.item()}, target_len={target_lengths.item()}, logits_T={student_logits.shape[1]}")

print("Labels:", labels)
print("Max Label Value:", labels.max().item(), "Pad ID:", processor.tokenizer.pad_token_id)
```

```
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import gc

# Clear memory
gc.collect()
torch.cuda.empty_cache()

# Define hyperparameters
lambda_distill = 0.7
max_label_len = 96  # For safe CTC alignment

# Optimizer
optimizer = AdamW(student_model.parameters(), lr=1e-4)

# Initialize projection layer later
whisper_proj_head = None

# Training loop
for step, batch in enumerate(tqdm(dataloader)):
    # Skip bad samples
    if batch is None or batch["input_values"] is None or batch["labels"] is None:
        print(f"[Step {step}] Skipped: Null batch")
        continue

    if len(batch["input_values"]) == 0 or len(batch["labels"]) == 0:
        print(f"[Step {step}] Skipped: Empty input or label")
        continue

    # Convert to tensors
    input_values = torch.tensor(batch["input_values"]).to(torch.float32).to("cuda")  # [T]
    labels = torch.tensor(batch["labels"]).to("cuda")                                # [L]

    print(f"[Step {step}] labels shape before crop: {labels.shape}")

    # Truncate labels to safe max length
    labels = labels[:max_label_len].unsqueeze(0)           # [1, L]
    input_values = input_values.unsqueeze(0)               # [1, T]

    # === Whisper teacher forward ===
    with torch.no_grad():
        whisper_inputs = teacher_processor.feature_extractor(
            input_values.float().cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to("cuda")

        whisper_outputs = teacher_model.encoder(whisper_inputs)
        whisper_logits = whisper_outputs.last_hidden_state  # [1, T_whisper, D]

    # === Student forward ===
    student_outputs = student_model(input_values.float())
    student_logits = student_outputs.logits  # [1, T_student, Vocab]

    # === Input & label length checks ===
    input_lengths = torch.tensor([student_logits.shape[1]], dtype=torch.long).to("cuda")
    target_lengths = torch.tensor([labels.shape[1]], dtype=torch.long).to("cuda")

    if input_lengths.item() < target_lengths.item():
        print(f"[Step {step}] Skipped: input_len={input_lengths.item()} < target_len={target_lengths.item()}")
        continue

    print(f"[Step {step}] input_len={input_lengths.item()}, target_len={target_lengths.item()}, logits_T={student_logits.shape[1]}")
    print("Labels:", labels)
    print("Max Label Value:", labels.max().item(), "Pad ID:", processor.tokenizer.pad_token_id)

    # === CTC Loss ===
    try:
        student_logits_ctc = student_logits.transpose(0, 1)  # [T, B, V]
        ctc_loss = F.ctc_loss(
            student_logits_ctc.log_softmax(dim=-1),
            labels,
            input_lengths,
            target_lengths,
            blank=processor.tokenizer.pad_token_id,
            reduction='mean',
            zero_infinity=True
        )
    except Exception as e:
        print(f"[Step {step}] CTC loss error: {str(e)}")
        continue

    # === Projection from Whisper to Student vocab ===
    if whisper_proj_head is None:
        whisper_proj_head = torch.nn.Linear(whisper_logits.size(-1), student_logits.size(-1)).to("cuda")

    whisper_logits_proj = whisper_proj_head(whisper_logits)  # [1, T, V]
    whisper_logits_interp = F.interpolate(
        whisper_logits_proj.transpose(1, 2),
        size=student_logits.shape[1],
        mode="linear"
    ).transpose(1, 2)

    # === KL Divergence Distillation Loss ===
    distill_loss = F.kl_div(
        student_logits.log_softmax(dim=-1),
        whisper_logits_interp.softmax(dim=-1),
        reduction="batchmean"
    )

    if torch.isnan(distill_loss) or torch.isnan(ctc_loss):
        print(f"[Step {step}] NaN Loss Detected. Skipped.")
        continue

    # === Backpropagation ===
    total_loss = ctc_loss + lambda_distill * distill_loss
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"[Step {step}] CTC Loss: {ctc_loss.item():.4f} | Distill Loss: {distill_loss.item():.4f}")

print("QLoRA + Whisper Distillation (CTC + KL) training complete.")
```

# 7. Evaluation

```
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
student_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english").to("cuda")
```

```
import torch
import torchaudio
import time
import numpy as np
from jiwer import wer
from psutil import Process
from datasets import load_dataset, Audio

# ============ Step 1: Load Test Data ============
fleurs_test = load_dataset("google/fleurs", "en_us", split="test[:1%]").cast_column("audio", Audio(sampling_rate=16000))
dns_noise = load_dataset("lj_speech", split="train[:1%]").cast_column("audio", Audio(sampling_rate=16000))

# ============ Step 2: Metric Containers ============
hyp_clean, ref_clean = [], []
hyp_noisy, ref_noisy = [], []
total_time, total_samples = 0.0, 0
process = Process()

# ============ Step 3: Define Noise Addition ============
def add_noise(speech, noise, snr_db):
    speech = np.array(speech)
    noise = np.array(noise)

    # Repeat or truncate noise to match speech length
    if len(noise) < len(speech):
        repeat_count = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeat_count)[:len(speech)]
    else:
        noise = noise[:len(speech)]

    # Compute power and apply SNR scaling
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    factor = (speech_power / noise_power) / (10 ** (snr_db / 10))
    noisy = speech + np.sqrt(factor) * noise
    return np.clip(noisy, -1.0, 1.0)

# ============ Step 4: Evaluation Loop ============
num_samples = min(6, len(fleurs_test))  # ensure valid indexing
for sample in fleurs_test.select(range(num_samples)):
    ref_text = sample["transcription"]
    speech = sample["audio"]["array"]
    noise = dns_noise[0]["audio"]["array"]

    # --- Clean Speech ---
    start_time = time.time()
    inputs = processor(speech, return_tensors="pt", sampling_rate=16000, padding=True).to("cuda")
    with torch.no_grad():
        logits = student_model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
    hyp_clean.append(transcription)
    ref_clean.append(ref_text)
    elapsed = time.time() - start_time
    total_time += elapsed
    total_samples += len(speech)

    # --- Noisy Speech ---
    noisy_speech = add_noise(speech, noise, snr_db=5)
    inputs_noisy = processor(noisy_speech, return_tensors="pt", sampling_rate=16000, padding=True).to("cuda")
    with torch.no_grad():
        logits_noisy = student_model(**inputs_noisy).logits
    pred_ids_noisy = torch.argmax(logits_noisy, dim=-1)
    transcription_noisy = processor.batch_decode(pred_ids_noisy)[0]
    hyp_noisy.append(transcription_noisy)
    ref_noisy.append(ref_text)

# ============ Step 5: Compute Metrics ============
wer_clean = wer(ref_clean, hyp_clean)
wer_noisy = wer(ref_noisy, hyp_noisy)
rtf = total_time / (total_samples / 16000)
mem_usage = process.memory_info().rss / 1024**2  # MB

print("=== Evaluation Results for DQLoRA ===")
print(f"WER (Clean): {wer_clean * 100:.2f}%")
print(f"WER (Noisy): {wer_noisy * 100:.2f}%")
print(f"RTF: {rtf:.3f}")
print(f"Memory Usage (MB): {mem_usage:.1f}")
```


<br><br>


## Citation

```bib
@article{yiruyang2025,
  title={A Lightweight Domain-Aware Denoising ASR via Adapter-guided Distillation},
  author={Yiru Yang, xxxx},
  journal={arXiv preprint arXiv:2505.xxxxx},
  year={2025}
}
```

<br><br><br><br>
