# CS5340 Project
Supervised fine-tuning code for CS5304 project

## Setup
```bash
conda create -n cs5340 python=3.10
conda activate cs5340
pip install -r requirements.txt
```

## Available Models
* [LLama2-7B](https://huggingface.co/magicgh/lima-llama2_7b)
* [Gemma-2B](https://huggingface.co/magicgh/lima-gemma_2b)
* [Mamba-1.4B](https://huggingface.co/HenryCai/mamba-1.4b-hf-lima)
* [Mamba-2.8B](https://huggingface.co/HenryCai/mamba-2.8b-hf-lima)

## Parameter Description

The `train_args` directory stores configuration files for different training methods used by different models. The main parameter descriptions are as follows:

* `output_dir`: Training output directory, stores checkpoints, tokenizers, tensorboard, etc.
model_name_or_path: Local directory of the pre-trained model, or the model name on huggingface.

* `train_file`: Training dataset path. For LIMA dataset, the default is `data/train.jsonl`.

* `template_name`: The template name used during instruction fine-tuning. For specific template names, please refer to the `component/template.py` file.

* `num_train_epochs`: Number of training epochs. In our experiments, the default is 5.

* `tokenize_num_workers`: Number of threads for tokenization during pre-training, defaults to 10.

* `deepspeed`: Deepspeed training configuration file. We do not use deepspeed for training.

* `train_mode`: Training mode, `full`, `lora` or `qlora`, defaults to `qlora` in our experiments.

* `task_type`: Task type, pretrain, sft or dpo, defaults to sft.
* `per_device_train_batch_size`: Batch size per GPU.
gradient_accumulation_steps: Number of gradient accumulation steps. Global batch = num_gpus * per_device_train_batch_size * gradient_accumulation_steps. We set it to 1 in our experiments.
* `gradient_checkpointing`: If the memory is insufficient, you can turn it on. It saves memory at the expense of time. The model does not cache the activation state and performs two forward calculations to save memory.
* `learning_rate`: Learning rate. For full parameter fine-tuning, it is recommended to be smaller, 1e-5 or 5e-6.

* `max_seq_length`: Maximum length during training. Set according to your own device. The longer the length, the more memory it will take.
* `max_prompt_length`: Maximum length of prompt when performing DPO.
* `logging_steps`: The number of steps to log the training loss.
* `save_steps`: Number of steps to save a model.
* `save_total_limit`: Maximum number of checkpoints to be saved in the output_dir directory. If exceeded, the oldest one will be deleted.
* `lr_scheduler_type`: Learning rate change strategy.
* `warmup_steps`: Warm-up steps. The number of steps for the learning rate to increase to the specified value.  
* `optim`: Optimizer. If it is full parameter fine-tuning, it is recommended to use adamw_hf.
* `seed`: Random seed, used to reproduce experimental results.
* `fp16`: Use fp16 mixed precision. V100 is recommended to be turned on.
* `bf16`: Use bf16 mixed precision. A100 is recommended to be turned on.

The following parameters need to be set when using QLoRA for training:

* `lora_rank`: Rank of the QLoRA matrix. It is generally set to 8, 16, 32, 64, etc. In the QLoRA paper, the author set it to 64. The larger the number, the more parameters involved in the training, the better the general effect will be, but more memory is required.
* `lora_alpha`: Scaling factor in QLoRA. It is generally set to 16 or 32.
* `lora_dropout`: Dropout rate of LORA weights.
The parameters of deepspeed can be modified as needed.


## Start Training
Replace `{num_gpus}` with the number of GPUs:

### Full-scale SFT
```bash
deepspeed --num_gpus={num_gpus} train.py --train_args_file train_args/bloom-1b1-sft-full.json
```


### Single Card SFT with QLoRA
```bash
python train.py --train_args_file train_args/llama2-7b-sft-qlora.json
```


### Multi-card SFT with QLoRA
```bash
torchrun --nproc_per_node={num_gpus} train.py --train_args_file train_args/llama2-7b-sft-qlora.json
```

## Weight Merging
If LoRA or QLoRA is used for training, this project only saves the adapter weights and configuration files. The adapter weights need to be merged with the base model. The script can be found at `script/merge_lora.py`.


## Acknowledgement
* [Firefly](https://github.com/yangjianxin1/Firefly)
* [LIMA](https://huggingface.co/datasets/GAIR/lima)