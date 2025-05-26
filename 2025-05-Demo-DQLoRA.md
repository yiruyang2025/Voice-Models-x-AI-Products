
# DQLoRA: Adapter-Guided Distillation for Lightweight ASR

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
fleurs = load_dataset("google/fleurs", "en_us", split="train[:1%]")
dns = load_dataset("lj_speech", split="train[:1%]")  # Use LJSpeech to simulate DNS noise
fleurs = fleurs.cast_column("audio", Audio(sampling_rate=16000))
```


# 3. Load Student Model (Wav2Vec2 + QLoRA)
```
student_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", load_in_4bit=True, device_map="auto")
student_model = prepare_model_for_kbit_training(student_model)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["encoder.layers.*.attention"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CTC
)
student_model = get_peft_model(student_model, lora_cfg)
student_model.train()

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

