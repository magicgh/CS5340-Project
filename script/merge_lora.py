from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse

def merge_lora_to_base_model(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_name_or_path,
        trust_remote_code=True,
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, args.adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='google/gemma-2b-it')
    parser.add_argument('--adapter_name_or_path', type=str, default='./model_output/gemma-2b-sft-qlora')
    parser.add_argument('--save_path', type=str, default='checkpoints/gemma-2b-sft-qlora-merge')
    args = parser.parse_args()
    merge_lora_to_base_model(args)
