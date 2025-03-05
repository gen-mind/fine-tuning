#!/usr/bin/env python
import argparse
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model with QLoRA and DeepSpeed.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="qwen-base-model",  # replace with the actual model name or path
        help="Path to pre-trained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="your_dataset_here",  # replace with your dataset id or local file
        help="The name or path of the dataset to use for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_model",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for training.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="./deepspeed_config.json",
        help="Path to DeepSpeed configuration file.",
    )
    # LoRA-specific arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability.")
    return parser.parse_args()

def load_model_and_tokenizer(model_name_or_path: str):
    """
    Loads the pre-trained model and tokenizer.
    Uses 4-bit quantization with BitsAndBytes and prepares the model for int8 training.
    """
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training()
    return model, tokenizer

def setup_peft_model(model, lora_r: int, lora_alpha: int, lora_dropout: float):
    """
    Configures and applies LoRA (QLoRA) to the model.
    Adjust the `target_modules` list based on the model architecture.
    """
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],  # adjust target modules as needed
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model

def tokenize_function(examples, tokenizer, max_length: int):
    """Tokenizes texts with truncation."""
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

def load_and_prepare_dataset(dataset_name: str, tokenizer, max_length: int):
    """
    Loads a dataset from the Hugging Face hub or local path
    and applies tokenization.
    """
    # Replace with appropriate split names for your dataset.
    dataset = load_dataset(dataset_name)
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    return tokenized_dataset

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir: str,
                per_device_train_batch_size: int, num_train_epochs: int,
                deepspeed_config: str):
    """Sets up TrainingArguments and trains the model using Trainer."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        deepspeed=deepspeed_config,
        fp16=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def main():
    args = parse_args()
    # Load model and tokenizer with 4-bit quantization support.
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    # Apply LoRA fine-tuning to the model.
    model = setup_peft_model(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    # Load and tokenize dataset.
    dataset = load_and_prepare_dataset(args.dataset_name, tokenizer, args.max_length)

    # Expecting the dataset to have "train" and "validation" splits.
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("Dataset must have 'train' and 'validation' splits.")

    # Train the model using DeepSpeed.
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        deepspeed_config=args.deepspeed_config,
    )

if __name__ == '__main__':
    main()