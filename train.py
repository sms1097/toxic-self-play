import torch
from utils import build_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from peft import (
    get_peft_model,
    PeftConfig,
    PeftModel,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer
import os


def main():
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    peft_model_id = "sms1097/unsafe-Mistral-7B-v0.1"
    lora_dropout = 0.05
    lora_rank = 8
    lora_alpha = 2 * lora_rank
    output_dir = peft_model_id
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 2
    optim = "paged_adamw_32bit"
    save_strategy = "steps"
    save_steps = 10
    logging_steps = 10
    learning_rate = 2e-3
    max_grad_norm = 0.3  # Sets limit for gradient clipping
    max_steps = 50  # Number of training steps
    warmup_ratio = 0.03  # Portion of steps used for learning_rate to warmup from 0
    lr_scheduler_type = "cosine"
    sanity_check = False

    ds = build_dataset(sanity_check=sanity_check)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias="none",  # setting to 'none' for only training weight params instead of biases
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        push_to_hub=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        # peft_model,
        peft_model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    peft_model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":
    main()
