import os
os.environ['TRANSFORMERS_CACHE'] = 'cache'

mask_time_prob = 0.01
epoch = 15
lr = 3e-4
os.system(f"python run_speech_recognition_ctc.py "
          "--dataset_name=./chr_voice/chr_voice.py "
          "--model_name_or_path=facebook/wav2vec2-large-xlsr-53 "
          "--dataset_config_name=chr "
          f"--output_dir=./wav2vec2-chr_voice-ep{epoch}-lr{lr}-mask{mask_time_prob} "
          "--overwrite_output_dir "
          f"--num_train_epochs={epoch} "
          "--per_device_train_batch_size=16 "
          "--gradient_accumulation_steps=2 "
          f"--learning_rate={lr} "
          "--warmup_steps=100 "
          "--evaluation_strategy=steps "
          "--audio_column_name=path "
          "--text_column_name=sentence "
          "--save_steps=100 "
          "--eval_steps=50 "
          "--layerdrop=0.0 "
          "--save_total_limit=1 "
          "--freeze_feature_extractor "
          "--gradient_checkpointing "
          "--chars_to_ignore , ? . ! - "
          f"--fp16 "
          f"--group_by_length --mask_time_prob {mask_time_prob} "
          f"--do_train --do_eval ")
