# SpeechProcessing-25

For Edge AI

<br><br>

## Setup Colab & Google Drive

xxxx:
```bash
# Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Project root on Drive
PROJ = "/content/drive/MyDrive/hearing_asr_dqlora"
!mkdir -p "$PROJ"
%cd "$PROJ"
print(f"Working directory -> {PROJ}")

# Core libraries
!pip install -q "transformers==4.40.2" "datasets[audio]" \
               "peft==0.10.0" "bitsandbytes" "accelerate" \
               "evaluate" "jiwer" "torchaudio"
```

<br><br>


## Clone speech-pretraining examples

xxxx:
```bash
git clone --depth 1 https://github.com/huggingface/transformers.git
cd transformers/examples/pytorch/speech-pretraining

# (Optional) patch helper scripts so they write to Drive by default
patch <<'EOF'
*** a/run_ctc.py
@@
- output_dir=args.output_dir,
+ output_dir=args.output_dir,  # already points to $PROJ via CLI below
EOF
```

<br><br>

## Prepare LibriSpeech data (train-clean-100)

xxxx:
```bash
python prepare_librispeech.py \
  --dataset_name "librispeech_asr" \
  --subset "train.clean.100" \
  --output_dir "$PROJ/data/librispeech"
```

<br><br>

## Pre-train teacher wav2vec 2.0 (CTC)

xxxx:
```bash
python run_wav2vec2_pretraining_no_trainer.py \
  --dataset_config_name "train.clean.100" \
  --dataset_cache_dir "$PROJ/cache" \
  --output_dir "$PROJ/models/teacher" \
  --num_train_epochs 1 \                    # demo: short run
  --per_device_train_batch_size 8 \
  --learning_rate 1e-4 \
  --logging_steps 100 \
  --save_steps 500 \
  --fp16
```

<br><br>

## Dump teacher logits  (on noisy / enhanced wavs)

xxxx:
```bash
python dump_logits.py \
  --model_name_or_path "$PROJ/models/teacher" \
  --audio_dir "$PROJ/data/librispeech" \
  --logit_cache "$PROJ/cache/teacher_logits" \
  --chunk_seconds 20
```

<br><br>

## Training adapters - “DQLoRA” Distillation

xxxx:
```bash
python run_ctc_adapter_distill.py \
  --teacher_logits "$PROJ/cache/teacher_logits" \
  --output_dir "$PROJ/models/student_dqlora" \
  --dataset_config_name "train.clean.100" \
  --dataset_cache_dir "$PROJ/cache" \
  --quant_bits 4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --logging_steps 50 --evaluation_strategy "steps" \
  --save_steps 250 --fp16
```

<br><br>

## Fine-tuning student with CTC Loss

xxxx:
```bash
python run_ctc_adapter_distill.py \
  --do_train --do_eval \
  --model_name_or_path "$PROJ/models/student_dqlora" \
  --output_dir "$PROJ/models/student_finetuned" \
  --dataset_config_name "train.clean.100" \
  --dataset_cache_dir "$PROJ/cache" \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --logging_steps 50 --save_steps 250 --fp16
```

<br><br>

## Sampling with Adapters (Inference)

xxxx:
```bash
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
