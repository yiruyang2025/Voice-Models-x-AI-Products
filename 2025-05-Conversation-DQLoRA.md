1. Install Dependencies

```
!pip install -q transformers datasets torchaudio peft accelerate bitsandbytes jiwer psutil
```

2. Imports

```
import torch, torchaudio, numpy as np, psutil, time
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    WhisperProcessor, WhisperModel,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
!pip install jiwer
from jiwer import wer
```


3. Load Data

```
# Load Chinese call-center datasets
train_ds = load_dataset("MagicData/CallHome", split="train").cast_column("audio", Audio(16000))
train_ds = train_ds.concatenate(load_dataset("aishell2", split="train").cast_column("audio", Audio(16000)))
test_clean = load_dataset("MagicData/CallHome", split="test[:1%]").cast_column("audio", Audio(16000))
test_noisy = test_clean  # we will add noise at eval
```



4. Prepare Models

```
device = "cuda"

# Teacher
whisper = WhisperModel.from_pretrained("openai/whisper-small").to(device).eval()
whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small")

# Student backbone + QLoRA Adapters
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
student = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    device_map="auto", quantization_config=bnb_cfg
)
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    target_modules=[f"wav2vec2.encoder.layers.{i}.{p}" 
                    for i in range(12) for p in ("attention.q_proj","attention.k_proj","attention.v_proj","attention.out_proj")]
)
student = get_peft_model(student, lora_cfg).to(device)
proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
```


5. Data Collation & Noise Augmentation

```
import random

# Load one DNS noise example
dns = load_dataset("microsoft/dns_challenge", split="train[:1%]").cast_column("audio", Audio(16000))[0]["audio"]["array"]

def add_noise(signal, snr_db):
    noise = np.resize(dns, signal.shape)
    sig_pow = np.mean(signal**2)
    noise_pow = np.mean(noise**2)
    factor = (sig_pow / noise_pow) / (10**(snr_db/10))
    return signal + np.sqrt(factor) * noise

def collate(batch):
    clean = batch["audio"]["array"]
    noisy = add_noise(np.array(clean), snr_db=random.uniform(0,10))
    inputs = proc(noisy, sampling_rate=16000, return_tensors="pt", padding=True)
    labels = proc(clean, sampling_rate=16000, return_tensors="pt", padding=True).input_ids
    inputs["labels"] = labels
    return {k:v.squeeze(0).to(device) for k,v in inputs.items()}

train_loader = DataLoader(train_ds, batch_size=4, collate_fn=collate)
```

6. Training Loop

```
optimizer = AdamW(student.parameters(), lr=1e-4)
lambda_kl = 0.7

# projection head from Whisper dim to voc size
proj = None

for epoch in range(3):
    for batch in train_loader:
        input_values = batch["input_values"].unsqueeze(0) if batch["input_values"].ndim==1 else batch["input_values"]
        labels = batch["labels"]
        
        # Teacher forward
        with torch.no_grad():
            t_feats = whisper.get_encoder()(
                whisper_proc.feature_extractor(input_values.cpu().numpy(), return_tensors="pt").input_features.to(device)
            ).last_hidden_state
        
        # Student forward
        s_out = student(input_values).logits  # [B, T, V]
        
        # CTC loss
        input_len = torch.tensor([s_out.shape[1]]).to(device)
        target_len = torch.tensor([labels.shape[1]]).to(device)
        ctc = F.ctc_loss(
            s_out.log_softmax(-1).transpose(0,1), labels, 
            input_len, target_len,
            blank=proc.tokenizer.pad_token_id, zero_infinity=True
        )
        
        # KL Distillation
        if proj is None:
            proj = torch.nn.Linear(t_feats.size(-1), s_out.size(-1)).to(device)
        t_logits = F.interpolate(
            proj(t_feats).transpose(1,2),
            size=s_out.size(1),
            mode="linear"
        ).transpose(1,2)
        kl = F.kl_div(s_out.log_softmax(-1), t_logits.softmax(-1), reduction="batchmean")
        
        loss = ctc + lambda_kl * kl
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

7. Evaluation


```
def evaluate(dataset, noisy=False):
    refs, hyps = [], []
    total_time = 0
    for sample in dataset:
        speech = sample["audio"]["array"]
        if noisy:
            speech = add_noise(speech, snr_db=5)
        start = time.time()
        inp = proc(speech, return_tensors="pt", sampling_rate=16000, padding=True).to(device)
        with torch.no_grad():
            pred = student(**inp).logits
        text = proc.batch_decode(pred.argmax(-1))[0]
        total_time += time.time() - start
        hyps.append(text)
        refs.append(sample["transcription"])
    return wer(refs, hyps), total_time / len(refs) / (len(speech)/16000)

wer_clean, rtf_clean = evaluate(test_clean, noisy=False)
wer_noisy, rtf_noisy = evaluate(test_noisy, noisy=True)
print(f"Clean WER: {wer_clean*100:.2f}%, RTF: {rtf_clean:.3f}")
print(f"Noisy WER: {wer_noisy*100:.2f}%, RTF: {rtf_noisy:.3f}")
```



<br>
