accelerate launch \
  code/open_models/finetune/sample_finetune_vision.py \
  --model_name_or_path microsoft/Phi-4-multimodal-instruct \
  --use_flash_attention \
  --full_run \
  --output_dir xxx \
  --batch_size_per_gpu 2 \
  --batch_size 128 \
  --num_train_epochs 3