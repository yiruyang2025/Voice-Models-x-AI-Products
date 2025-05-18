# SpeechProcessing-25

<br>

- For Edge AI, with NVIDIA A100 Tensor Core GPU

<br>

# Option 1 - [facebook/wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)

<br>

- [2020 - wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)

<br>

```bash
@inproceedings{baevski2020wav2vec,
  title        = {wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  author       = {Baevski, Alexei and Zhou, Henry and Mohamed, Abdelrahman and Auli, Michael},
  booktitle    = {Advances in Neural Information Processing Systems (NeurIPS)},
  year         = {2020},
  url          = {https://arxiv.org/abs/2006.11477}
}
```

<br>

- **wav2vec2-large-960h-lv60-self** is based on the above papaer, 60k hours of unlabeled Libri-Light data were used for self-supervised pre-training and fine-tuned on LibriSpeech 960h<br>

<br><br>

## 1. Setup Colab & Google Drive

<br>

```bash
from google.colab import drive
import os

if not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")

PROJECT_ROOT="/content/drive/MyDrive/hearing_asr_dqlora"
os.makedirs(PROJECT_ROOT, exist_ok=True)
%cd $PROJECT_ROOT
echo "Working directory: $PROJECT_ROOT"
```

<br><br>

```bash
%%bash
pip uninstall -y -q sentence-transformers
pip install -q transformers==4.40.2 "datasets[audio]" peft==0.10.0 bitsandbytes accelerate evaluate jiwer torchaudio

export PROJECT_ROOT="/content/drive/MyDrive/hearing_asr_dqlora"
export HF_HOME="$PROJECT_ROOT/cache/hf"
export HF_DATASETS_CACHE="$PROJECT_ROOT/cache/hf"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/cache/hf"
export TMPDIR="$PROJECT_ROOT/cache/tmp"

mkdir -p "$HF_HOME" "$TMPDIR"
echo "Hugging Face cache directory: $HF_HOME"
```


<br><br>

## 2. Pretrained model – facebook/wav2vec2-large-960h-lv60-self

<br>

```bash
%%bash
pip uninstall -y torch torchvision torchaudio fastai
pip install -q torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124
```

<br><br>


```bash
%%bash
set -euxo pipefail
export TORCHDYNAMO_DISABLE=1

python3 - << 'EOF'
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

PROJECT_ROOT="/content/drive/MyDrive/hearing_asr_dqlora"
TEACHER_DIR=os.path.join(PROJECT_ROOT,"models","teacher")
os.makedirs(TEACHER_DIR,exist_ok=True)

MODEL_ID="facebook/wav2vec2-large-960h-lv60-self"
model=Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
processor=Wav2Vec2Processor.from_pretrained(MODEL_ID)

model.save_pretrained(TEACHER_DIR)
processor.save_pretrained(TEACHER_DIR)

print(f"Saved pretrained model and processor to {TEACHER_DIR}")
EOF
```

<br><br>


## 3. Dataset for Fine-tuning (LibriSpeech train-clean-100 / validation / test)

<br>

```bash
# ===== 3A. Mount Google Drive and download LibriSpeech to persistent storage =====

# 1) Mount (or remount) Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 2) Define project and cache paths on Drive
import os
PROJECT_ROOT = "/content/drive/MyDrive/hearing_asr_dqlora"
HF_CACHE     = os.path.join(PROJECT_ROOT, "cache", "hf")
os.makedirs(HF_CACHE, exist_ok=True)

# 3) Download and extract the LibriSpeech ASR "clean" subset
from datasets import load_dataset_builder, DownloadConfig

builder = load_dataset_builder("librispeech_asr", "clean")
download_config = DownloadConfig(cache_dir=HF_CACHE)
builder.download_and_prepare(download_config=download_config)

print("✔ Download and extraction to Google Drive completed.")
```

<br><br>


```bash
%%bash
# ===== 3B. Convert to Arrow format with inline progress bars and save to Google Drive =====
set -euxo pipefail

export PROJECT_ROOT="/content/drive/MyDrive/hearing_asr_dqlora"
export HF_HOME="${PROJECT_ROOT}/cache/hf"
```

<br><br>


```bash
# A1_download_extract.py
# 1) Mount Drive (for final write-back)
# 2) Download & extract LibriSpeech locally under /content
# 3) Copy extracted folder to Drive for permanence

import os
import shutil
from datasets import load_dataset_builder, DownloadConfig
from google.colab import drive

# 1. Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# 2. Local paths
LOCAL_CACHE = "/content/hf_cache"
os.makedirs(LOCAL_CACHE, exist_ok=True)

# 3. Download & extract into local cache
dl_cfg  = DownloadConfig(cache_dir=LOCAL_CACHE, force_download=False)
builder = load_dataset_builder("librispeech_asr", "clean", cache_dir=LOCAL_CACHE)
builder.download_and_prepare(download_config=dl_cfg)

# 4. Copy extracted files to Drive
DRIVE_EXTRACT = "/content/drive/MyDrive/hearing_asr_dqlora/cache/hf/librispeech_asr/clean"
if os.path.isdir(DRIVE_EXTRACT):
    shutil.rmtree(DRIVE_EXTRACT)
shutil.copytree(os.path.join(LOCAL_CACHE, "librispeech_asr", "clean"), DRIVE_EXTRACT)

print("✔ LibriSpeech extracted locally and copied to Drive at:", DRIVE_EXTRACT)
```

<br><br>


```bash
!rm -rf /content/hf_cache
!rm -rf /content/librispeech_arrow
!rm -rf /content/tmp_arrows
!rm -rf /content/local_librispeech_clean
```

<br><br>


```bash
!df -h /content
```

<br><br>


```bash
# A2_split_and_save.py
# 1) Mount Google Drive
# 2) Load each LibriSpeech split from the Drive cache without redownloading
# 3) Prune columns and save Arrow datasets back to Drive

import os
from datasets import load_dataset, Audio
from google.colab import drive

# 1. Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# 2. Define paths on Drive
DRIVE_CACHE_DIR = "/content/drive/MyDrive/hearing_asr_dqlora/cache/hf"
DRIVE_OUTPUT    = "/content/drive/MyDrive/hearing_asr_dqlora/data/librispeech_arrow"

os.makedirs(DRIVE_CACHE_DIR, exist_ok=True)
os.makedirs(DRIVE_OUTPUT, exist_ok=True)

# 3. Generate, prune, and save each split
for hf_split, folder in [
    ("train.100",  "train100"),
    ("validation", "validation"),
    ("test",       "test")
]:
    output_path = os.path.join(DRIVE_OUTPUT, folder)
    if os.path.isdir(output_path):
        continue

    # Load split, reusing existing cache and never redownloading
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split=hf_split,
        cache_dir=DRIVE_CACHE_DIR,
        download_mode="reuse_cache_if_exists"
    )

    # Cast audio column and remove all but audio & text
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    cols_to_drop = [c for c in ds.column_names if c not in ("audio", "text")]
    ds = ds.remove_columns(cols_to_drop)

    # Save Arrow dataset directly to Drive
    ds.save_to_disk(output_path)
    print(f"Saved split '{folder}' to {output_path}")
```

<br><br>


```bash
# A3_generate_splits.py
import os, shutil
from tqdm import tqdm
from datasets import load_dataset, Audio
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

PROJECT = "/content/drive/MyDrive/hearing_asr_dqlora"
CACHE   = os.path.join(PROJECT, "cache", "hf")
TMP     = os.path.join(PROJECT, "cache", "tmp_arrows")
FINAL   = os.path.join(PROJECT, "data", "librispeech_arrow")
os.makedirs(TMP, exist_ok=True)
os.makedirs(FINAL, exist_ok=True)

os.environ["HF_HOME"]           = CACHE
os.environ["HF_DATASETS_CACHE"] = CACHE
os.environ["TRANSFORMERS_CACHE"]= CACHE

for split,path in tqdm({"train.100":"train100","validation":"validation","test":"test"}.items()):
    out = os.path.join(FINAL, path)
    if os.path.isdir(out): continue
    ds = load_dataset("librispeech_asr","clean",split=split,cache_dir=CACHE,local_files_only=True)\
         .cast_column("audio",Audio(sampling_rate=16000))\
         .remove_columns([c for c in ds.column_names if c not in ("audio","text")])
    tmp = os.path.join(TMP, path)
    ds.save_to_disk(tmp)
    shutil.move(tmp, out)
    print("✔", path, "->", out)
```

<br><br>



## 4. Dump teacher logits (on noisy / enhanced wavs) to Drive

<br>

```bash
# 4. Dump teacher logits
cd "$PROJECT_ROOT"
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
# 5. Adapter distillation on first 50 hours + early stopping
cd "$PROJECT_ROOT"
python run_ctc_adapter_distill.py \
  --teacher_logits cache/teacher_logits \
  --output_dir models/student_dqlora_subset \
  --dataset_config_name clean \
  --train_split_name train100 \
  --validation_split_name validation \
  --data_dir data/librispeech_arrow \
  --dataset_cache_dir "$HF_HOME" \
  --quant_bits 4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_duration_hours 50 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --load_best_model_at_end True \
  --metric_for_best_model wer \
  --greater_is_better False \
  --save_total_limit 3 \
  --logging_steps 50 \
  --save_steps 500 \
  --fp16
```

<br><br>

## 6. Fine-tune student with CTC Loss

<br>

```bash
# 6. Fine-tune student with CTC Loss + early stopping
cd "$PROJECT_ROOT"
python run_ctc_adapter_distill.py \
  --do_train --do_eval \
  --model_name_or_path models/student_dqlora_subset \
  --output_dir models/student_finetuned_subset \
  --dataset_config_name clean \
  --train_split_name train100 \
  --validation_split_name validation \
  --data_dir data/librispeech_arrow \
  --dataset_cache_dir "$HF_HOME" \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --load_best_model_at_end True \
  --metric_for_best_model wer \
  --greater_is_better False \
  --save_total_limit 2 \
  --logging_steps 50 \
  --save_steps 200 \
  --fp16
```

<br><br>


## 7. Sampling with Adapters (Inference) 

<br>

```bash
# 7. Inference
cd "$PROJECT_ROOT"
python transcribe.py \
  --model_path models/student_finetuned_subset \
  --audio_file "/content/drive/MyDrive/your_test_audio.wav" \
  --chunk_length_s 30 \
  --stride_length_s 5
```


<br><br><br><br>


# Option 2 - Pre-train By Yourself

## 1. Setup Colab & Google Drive

<br>

```bash
from google.colab import drive
import os

# 1) Mount Google Drive if not already mounted
if not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")

# 2) Define project root on Drive
PROJECT_ROOT = "/content/drive/MyDrive/hearing_asr_dqlora"
os.makedirs(PROJECT_ROOT, exist_ok=True)

# 3) Change to the project directory
%cd $PROJECT_ROOT
print("Working directory:", PROJECT_ROOT)
```

<br><br>


```bash
%%bash
# 4) Uninstall potential conflicting package and install required libraries
pip uninstall -y -q sentence-transformers
pip install -q transformers==4.40.2 "datasets[audio]" peft==0.10.0 bitsandbytes accelerate evaluate jiwer torchaudio

# 5) Configure Hugging Face caches on Drive
export PROJECT_ROOT="/content/drive/MyDrive/hearing_asr_dqlora"
export HF_HOME="$PROJECT_ROOT/cache/hf"
export HF_DATASETS_CACHE="$PROJECT_ROOT/cache/hf"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/cache/hf"
export TMPDIR="$PROJECT_ROOT/cache/tmp"

# 6) Create cache directories
mkdir -p "$HF_HOME" "$TMPDIR"

echo "Hugging Face cache directory: $HF_HOME"
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
  title={XXX-Only Distillation for Hearing Aid ASR},
  author={Yiru Yang, xxxx},
  journal={arXiv preprint arXiv:2506.xxxxx},
  year={2025}
}
```

<br><br>
