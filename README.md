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
# prerequisites: Sections 1–2 executed

import os, time, shutil
from pathlib import Path
from datasets import load_dataset, disable_caching, Audio

# project root on Drive (same path you used before)
PROJ = "/content/drive/MyDrive/hearing_asr_dqlora"
DATA_DIR   = Path(f"{PROJ}/data/librispeech_100")    # final Arrow files (Drive)
TMP_CACHE  = Path("/tmp/hf_cache")                   # local cache, auto-cleared

DATA_DIR.mkdir(parents=True, exist_ok=True)
TMP_CACHE.mkdir(parents=True, exist_ok=True)

# point HF caches to /tmp (fast, avoids Drive-FUSE limitation)
os.environ["HF_HOME"]            = str(TMP_CACHE)
os.environ["HF_DATASETS_CACHE"]  = str(TMP_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(TMP_CACHE)

disable_caching()  # always use cache_dir parameter
print("HF cache   :", TMP_CACHE)
print("Arrow data :", DATA_DIR)

splits = {
    "train.100" : "train.clean.100",
    "validation": "validation.clean",
    "test"      : "test.clean"
}

for name, hf_split in splits.items():
    t0 = time.time()
    print(f"\n▶ preparing split {name}")

    ds = (
        load_dataset(
            "librispeech_asr",
            "clean",
            split=hf_split,
            cache_dir=TMP_CACHE,
            streaming=False           # download fully to /tmp
        )
        .cast_column("audio", Audio(sampling_rate=16_000))
    )

    out_path = DATA_DIR / name
    ds.save_to_disk(out_path)         # write Arrow files to Drive
    print(f"✓ saved → {out_path} | rows={len(ds):,} | {time.time()-t0:.1f}s")

# remove /tmp cache to free Colab disk
shutil.rmtree(TMP_CACHE, ignore_errors=True)
print("\nLibriSpeech ready at", DATA_DIR)
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
