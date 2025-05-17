# SpeechProcessing-25

- For Edge AI, with NVIDIA A100 Tensor Core GPU

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

/content/drive/MyDrive/hearing_asr_dqlora/data/librispeech_raw/
    ├── train-clean-100/
    ├── dev-clean/
    └── test-clean/

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



```

<br><br>


```bash



```

<br><br>



```bash



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
