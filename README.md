# SpeechProcessing-25

<br>

- For Edge AI, with NVIDIA A100 Tensor Core GPU

<br>

# Option 1 - [facebook/wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) by API

<br>

- [2020 - wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)

<br>

- **wav2vec2-large-960h-lv60-self** is based on the above papaer, 60k hours of unlabeled Libri-Light data were used for self-supervised pre-training and fine-tuned on LibriSpeech 960h<br>

<br><br>

## 1. Setup Colab & Google Drive

<br>

```bash
# Mount Drive and install dependencies
from google.colab import drive
drive.mount("/content/drive")

# Define project root
export PROJ="/content/drive/MyDrive/hearing_asr_dqlora"
mkdir -p "$PROJ"/{cache/hf,data,models}

# Install core libraries
pip install -q transformers datasets[audio] peft bitsandbytes accelerate torchaudio jiwer
```

```bash
# Redirect HF caches to Drive
export HF_HOME="$PROJ/cache/hf"
export HF_DATASETS_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
echo "HF cache dir: $HF_HOME"
```


<br><br>

## 2. Pretrained model – facebook/wav2vec2-large-960h-lv60-self

<br>

```bash
python3 - << 'EOF'
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import os

model_name = "facebook/wav2vec2-large-960h-lv60-self"
out_dir = os.path.join(os.environ["PROJ"], "models/teacher")

# Load and save
model = Wav2Vec2ForCTC.from_pretrained(model_name)
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
EOF
```

<br><br>


## 3. Dataset for Fine-tuning (LibriSpeech train-clean-100 / validation / test)

<br>

```bash
python3 - << 'EOF'
from datasets import load_dataset, Audio
import os

proj = os.environ["PROJ"]
hf_cache = os.environ["HF_HOME"]
data_dir = os.path.join(proj, "data/librispeech_arrow")
os.makedirs(data_dir, exist_ok=True)

splits = {
    "train.clean.100": "train100",
    "validation.clean": "validation",
    "test.clean"      : "test"
}

for hf_split, name in splits.items():
    out_path = os.path.join(data_dir, name)
    if os.path.isdir(out_path):
        print(f"Skip {name}, exists.")
        continue
    print(f"Loading {hf_split} …")
    ds = load_dataset("librispeech_asr", "clean", split=hf_split,
                      cache_dir=hf_cache)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds.save_to_disk(out_path)
    print(f"Saved {name}: {len(ds)} examples")
EOF
```

<br><br>



## 4. Dump teacher logits (on noisy / enhanced wavs) to Drive

<br>

```bash
cd "$PROJ"
python dump_logits.py \
  --model_name_or_path models/teacher \
  --audio_dir data/librispeech_arrow/train100 \
  --logit_cache cache/teacher_logits \
  --chunk_seconds 20
```

<br><br>

## 5. Training adapters – DQLoRA distillation

<br>

```bash
cd "$PROJ"
python run_ctc_adapter_distill.py \
  --teacher_logits cache/teacher_logits \
  --output_dir models/student_dqlora \
  --dataset_config_name clean \
  --train_split_name train100 \
  --validation_split_name validation \
  --dataset_cache_dir "$HF_HOME" \
  --data_dir data/librispeech_arrow \
  --quant_bits 4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --logging_steps 50 \
  --save_steps 250 \
  --fp16
```

<br><br>

## 6. Fine-tune student with CTC Loss

<br>

```bash
cd "$PROJ"
python run_ctc_adapter_distill.py \
  --do_train --do_eval \
  --model_name_or_path models/student_dqlora \
  --output_dir models/student_finetuned \
  --dataset_config_name clean \
  --train_split_name train100 \
  --validation_split_name validation \
  --dataset_cache_dir "$HF_HOME" \
  --data_dir data/librispeech_arrow \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --logging_steps 50 \
  --save_steps 250 \
  --fp16
```

<br><br>


## 7. Sampling with Adapters (Inference) 

<br>

```bash
cd "$PROJ"
python transcribe.py \
  --model_path models/student_finetuned \
  --audio_file "/content/drive/MyDrive/your_test_audio.wav" \
  --chunk_length_s 30 \
  --stride_length_s 5
```

<br><br>



<br><br><br><br>


# Option 2 - Pre-train By Yourself

## 1. Setup Colab & Google Drive

<br>

```bash
# ===== 1. Setup Colab & Google Drive =====
# Safe-mount: only mount if Drive is not yet available
from google.colab import drive
import os, pathlib, subprocess, time, json, shlex, textwrap

if not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")          # interactive one-time mount

# Project root on Drive
PROJ = "/content/drive/MyDrive/hearing_asr_dqlora"
os.makedirs(PROJ, exist_ok=True)
%cd "$PROJ"
print(f"Working directory -> {PROJ}")

# ------------------------------------------------------------------
# (optional) remove conflicting package, then install core libraries
# ------------------------------------------------------------------
%pip uninstall -y -q sentence-transformers        # optional conflict fix
!pip install -q "transformers==4.40.2" \
               "datasets[audio]" "peft==0.10.0" \
               "bitsandbytes" "accelerate" \
               "evaluate" "jiwer" "torchaudio"

# ------------------------------------------------------------------
# Hugging Face caches → Drive (permanent, frees Colab disk)
# ------------------------------------------------------------------
HF_CACHE = f"{PROJ}/cache/hf"
os.makedirs(HF_CACHE, exist_ok=True)

os.environ["HF_HOME"]            = HF_CACHE
os.environ["HF_DATASETS_CACHE"]  = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["TMPDIR"]             = f"{PROJ}/cache/tmp"

print("HF cache dir:", HF_CACHE)
```

<br><br>


## 2. Clone speech-pretraining examples

<br>

```bash
%%bash
# ===== 2. Clone speech-pretraining examples =====
set -e

REPO_DIR="$PROJ/transformers"

# Clone once; skip if it already exists
if [ ! -d "$REPO_DIR" ]; then
  git clone --depth 1 https://github.com/huggingface/transformers.git "$REPO_DIR"
fi

# Enter the speech-pretraining folder
cd "$REPO_DIR/examples/pytorch/speech-pretraining" || exit 1
pwd
```

<br><br>

## 3. Prepare LibriSpeech data (train-clean-100) to Drive

<br>

```bash
/content/drive/MyDrive/hearing_asr_dqlora/data/librispeech_raw/
    ├── train-clean-100/
    ├── dev-clean/
    └── test-clean/
```


<br><br>

```bash
# ===== 3-A1. Download LibriSpeech archives to Google Drive =====
# Prerequisites: Sections 1–2 executed (Drive mounted, PROJ defined)

import os
from pathlib import Path

# 1) Project base on Drive
PROJ        = "/content/drive/MyDrive/hearing_asr_dqlora"
ARCHIVE_DIR = Path(f"{PROJ}/cache/archives")
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# 2) URLs for the splits
urls = {
    "train-clean-100": "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "dev-clean"      : "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean"     : "http://www.openslr.org/resources/12/test-clean.tar.gz",
}

# 3) Download each archive into Drive
for name, url in urls.items():
    out_path = ARCHIVE_DIR / f"{name}.tar.gz"
    if out_path.exists():
        print(f"✓ {out_path.name} already on Drive")
    else:
        print(f"Downloading {out_path.name} …")
        # -c to resume partial downloads
        os.system(f"wget -c {url} -O {out_path}")

print("\nAll archives are in", ARCHIVE_DIR)
```


<br><br>

```bash
# ===== 3-A2. Extract LibriSpeech archives into Google Drive =====
# Prerequisites: 3-A1 completed

import tarfile
from pathlib import Path

PROJ     = "/content/drive/MyDrive/hearing_asr_dqlora"
ARCHIVE_DIR = Path(f"{PROJ}/cache/archives")
RAW_DIR     = Path(f"{PROJ}/data/librispeech_raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 1) Iterate over each .tar.gz in Drive and extract to RAW_DIR
for archive in ARCHIVE_DIR.glob("*.tar.gz"):
    split_name = archive.stem               # e.g. "train-clean-100"
    target_dir = RAW_DIR / split_name
    if target_dir.exists():
        print(f"✓ {split_name} already extracted")
        continue
    print(f"Extracting {archive.name} → {target_dir} …")
    with tarfile.open(archive) as tf:
        tf.extractall(path=RAW_DIR)

print("\nAll splits extracted under", RAW_DIR)
```

<br><br>


```bash
# ===== 3-B. Build Arrow datasets on Google Drive =====
from pathlib import Path
from datasets import load_dataset, Audio

PROJ      = "/content/drive/MyDrive/hearing_asr_dqlora"
RAW_DIR   = Path(PROJ, "data/librispeech_raw")
ARROW_DIR = Path(PROJ, "data/librispeech_arrow")
ARROW_DIR.mkdir(parents=True, exist_ok=True)

mapping = {
    "train-clean-100": "train.100",
    "dev-clean"      : "validation",
    "test-clean"     : "test",
}

for raw_split, hf_split in mapping.items():
    out_path = ARROW_DIR / hf_split
    if out_path.exists():
        print(f"✓ Arrow {hf_split} exists, skipping")
        continue
    print(f"Processing {hf_split} → {out_path}")
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split=hf_split,
        cache_dir=str(PROJ + "/cache/archives")
    ).cast_column("audio", Audio(sampling_rate=16_000))
    ds.save_to_disk(out_path)
    print(f"Saved {hf_split}: {len(ds)} examples")
```

<br><br>



## 4. Pre-train teacher wav2vec 2.0 (CTC) – demo epoch

<br>

```bash
%%bash
# ===== 4. Pre-train teacher wav2vec2 (CTC) – demo epoch with full logging =====

# 1) Define your Drive project root and example directory
PROJ="/content/drive/MyDrive/hearing_asr_dqlora"
EX_DIR="${PROJ}/transformers/examples/pytorch/speech-pretraining"

# 2) Fail fast on any error and disable W&B
set -euxo pipefail
export WANDB_DISABLED=true

# 3) Change into the example directory
cd "$EX_DIR"

# 4) Launch a short demo pre-training run
#    NOTE: logging_steps is low so you see frequent updates,
#    and we print all logs directly to the notebook (no filtering).
python run_wav2vec2_pretraining_no_trainer.py \
  --dataset_name librispeech_asr \
  --dataset_config_names clean \
  --dataset_split_names train.100 validation \
  --model_name_or_path facebook/wav2vec2-base \
  --cache_dir "${PROJ}/cache/hf" \
  --output_dir "${PROJ}/models/teacher" \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --logging_steps 5 \
  --saving_steps 500
```

<br><br>

## 5. Dump teacher logits (on noisy / enhanced wavs) to Drive

<br>

```bash
cd "$repo_dir"

python dump_logits.py \
  --model_name_or_path "$PROJ/models/teacher" \
  --audio_dir "$DATA_DIR" \
  --logit_cache "$PROJ/cache/teacher_logits" \
  --chunk_seconds 20
```

<br><br>

## 6. Training adapters – DQLoRA distillation

```bash
cd "$repo_dir"

python run_ctc_adapter_distill.py \
  --teacher_logits "$PROJ/cache/teacher_logits" \
  --output_dir "$PROJ/models/student_dqlora" \
  --dataset_config_name clean \
  --train_split_name train.100 \
  --validation_split_name validation \
  --dataset_cache_dir "$HF_CACHE" \
  --data_dir "$DATA_DIR" \
  --quant_bits 4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --logging_steps 50 --save_steps 250 --fp16
```

<br><br>

## 7. Fine-tune student with CTC Loss

```bash
cd "$repo_dir"

python run_ctc_adapter_distill.py \
  --do_train --do_eval \
  --model_name_or_path "$PROJ/models/student_dqlora" \
  --output_dir "$PROJ/models/student_finetuned" \
  --dataset_config_name clean \
  --train_split_name train.100 \
  --validation_split_name validation \
  --dataset_cache_dir "$HF_CACHE" \
  --data_dir "$DATA_DIR" \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --logging_steps 50 --save_steps 250 --fp16
```

<br><br>

## 8. Sampling with Adapters (Inference) 

```bash
cd "$repo_dir"

python transcribe.py \
  --model_path "$PROJ/models/student_finetuned" \
  --audio_file "/content/drive/MyDrive/your_test_audio.wav" \
  --chunk_length_s 30 --stride_length_s 5
```

<br><br><br><br>



## 9. Others - If needed

```bash
# ===== 0. Clean Google-Drive artefacts (datasets, cache, models) =====
# Run ONLY if you are sure you want to free all space taken by prior runs

from google.colab import drive
import shutil, os, textwrap, pathlib, time

drive.mount("/content/drive")             # 1) mount Drive

PROJ   = "/content/drive/MyDrive/hearing_asr_dqlora"
TARGETS = [
    f"{PROJ}/data",     # LibriSpeech Arrow files
    f"{PROJ}/cache",    # Hugging-Face download / extract cache
    f"{PROJ}/models"    # teacher / student checkpoints
]

print("\nFolders scheduled for removal:\n")
for p in TARGETS:
    print("  •", p)

# 2) one last confirmation
resp = input("\nType DELETE (all caps) to remove everything above: ")
if resp.strip() == "DELETE":
    t0 = time.time()
    for p in TARGETS:
        if os.path.exists(p):
            print(f"→ deleting {p} …", end="", flush=True)
            shutil.rmtree(p)
            print(" done")
        else:
            print(f"→ {p} not found – skipped")
    print(f"\n✓ clean-up finished  ({time.time()-t0:.1f}s)")
else:
    print("Aborted – nothing was deleted.")
```

<br><br><br><br>


## Citation

```bib
@article{yiruyang2025,
  title={Adapter-Only Distillation for Hearing Aid ASR},
  author={Yiru Yang, xxxx},
  journal={arXiv preprint arXiv:2506.xxxxx},
  year={2025}
}
```

<br><br>
