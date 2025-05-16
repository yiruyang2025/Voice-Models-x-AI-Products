# SpeechProcessing-25

For Edge AI

<br><br>

## 1. Setup Colab & Google Drive

<br>

```bash
# Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Project root on Drive
PROJ = "/content/drive/MyDrive/hearing_asr_dqlora"
!mkdir -p "$PROJ"
%cd "$PROJ"
print(f"Working directory -> {PROJ}")

# Remove potential conflicts and install core libs
%pip uninstall -y -q sentence-transformers         # optional
!pip install -q "transformers==4.40.2" \
               "datasets[audio]" "peft==0.10.0" \
               "bitsandbytes" "accelerate" \
               "evaluate" "jiwer" "torchaudio"

# Send every Hugging Face cache to Drive
import os, pathlib, textwrap, subprocess, shlex, time, json

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
# Shell cell
if [ ! -d "$PROJ/transformers" ]; then
  git clone --depth 1 https://github.com/huggingface/transformers.git "$PROJ/transformers"
fi
cd "$PROJ/transformers/examples/pytorch/speech-pretraining"
repo_dir="$PWD"
echo "Repo cloned at $repo_dir"
```

<br><br>

## 3. Prepare LibriSpeech data (train-clean-100) to Drive

<br>

```bash
# ===== 3. Prepare LibriSpeech data (train-clean-100) =====
# Run this cell after section 2 (repo cloned)

import os, time
from pathlib import Path
from datasets import load_dataset, Audio, disable_caching

PROJ = "/content/drive/MyDrive/hearing_asr_dqlora"
REPO_DIR = f"{PROJ}/transformers"                     #  <-- 与板块2保持一致
EXAMPLE_DIR = f"{REPO_DIR}/examples/pytorch/speech-pretraining"

# cd into example folder
%cd "$EXAMPLE_DIR"

# ---------------- directories on Google Drive ----------------
DATA_DIR  = Path(f"{PROJ}/data/librispeech_full")     # Arrow sets
# CACHE_DIR = Path(f"{PROJ}/cache/hf_datasets")         # HF download & extract # <-- Remove or comment out this line
TMP_DIR   = Path(f"{PROJ}/cache/tmp")                 # temp for tar extraction

for d in (DATA_DIR, TMP_DIR): # <-- Removed CACHE_DIR from this loop
    d.mkdir(parents=True, exist_ok=True)

# Create a temporary cache directory within the Colab environment's local filesystem
LOCAL_CACHE_DIR = Path("/tmp/hf_datasets_local_cache")
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# route temp path to Drive
# os.environ["HF_HOME"]            = str(CACHE_DIR) # <-- Remove or comment out this line
# os.environ["HF_DATASETS_CACHE"]  = str(CACHE_DIR) # <-- Remove or comment out this line
# os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR) # <-- Remove or comment out this line

# Set HF cache to the local temporary directory for downloading
os.environ["HF_HOME"]            = str(LOCAL_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"]  = str(LOCAL_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(LOCAL_CACHE_DIR)

os.environ["TMPDIR"]             = str(TMP_DIR) # Keep TMPDIR on Drive if needed for large temp files during processing

disable_caching()   # rely solely on explicit dirs above

print("HF_HOME =", os.environ['HF_HOME'])
print("TMPDIR  =", os.environ['TMPDIR'])
print("Downloading LibriSpeech splits: train.clean.100 / validation.clean / test.clean")

splits = {
    "train.100":  "train.clean.100",
    "validation": "validation.clean",
    "test":       "test.clean"
}

for local_name, hf_split in splits.items():
    t0 = time.time()
    print(f"\n▶  preparing split: {local_name}")

    # Download the dataset to the local temporary cache directory
    ds = load_dataset(
        "librispeech_asr",
        "clean",                 # builder config
        split=hf_split,
        cache_dir=LOCAL_CACHE_DIR, # Use the local temporary cache directory
        streaming=False          # full download to local temp
    ).cast_column("audio", Audio(sampling_rate=16_000))

    out_path = DATA_DIR / local_name
    # Save the processed dataset to the desired location on Google Drive
    ds.save_to_disk(out_path)
    print(f"✓ saved  {out_path}   rows={len(ds):,}   time={time.time()-t0:.1f}s")

print("\nLibriSpeech ready →", DATA_DIR)
```

<br><br>

## 4. Pre-train teacher wav2vec 2.0 (CTC) – demo epoch

<br>

```bash
cd "$repo_dir"

python run_wav2vec2_pretraining_no_trainer.py \
  --dataset_config_name clean \
  --train_split_name train.100 \
  --validation_split_name validation \
  --dataset_cache_dir "$HF_CACHE" \
  --data_dir "$DATA_DIR" \
  --output_dir "$PROJ/models/teacher" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-4 \
  --logging_steps 100 \
  --save_steps 500 \
  --fp16
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
