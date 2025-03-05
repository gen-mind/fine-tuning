import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments

# --- Model & 4-bit Quantization Setup ---
model_name = "Qwen/Qwen-7B-Chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the model in 4-bit mode with automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load the tokenizer; set pad_token if missing
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     model.resize_token_embeddings(len(tokenizer))


# --- Apply LoRA Fine-Tuning ---
# Configure LoRA to adapt key projection layers (adjust target modules if needed)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --- Define Alpaca Prompt Template ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

# EOS_TOKEN = tokenizer.eos_token  # Ensure EOS token is appended
EOS_TOKEN = tokenizer.eos_token if tokenizer.eos_token is not None else ""


# def formatting_prompts_func(examples):
#     instructions = examples["instruction"]
#     inputs = examples["input"]
#     outputs = examples["output"]
#     texts = []
#     for instruction, input_text, output in zip(instructions, inputs, outputs):
#         text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
#         texts.append(text)
#     return {"text": texts}
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Replace None with an empty string
        instruction = instruction if instruction is not None else ""
        input_text = input_text if input_text is not None else ""
        output = output if output is not None else ""
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


# # --- Load & Preprocess the Alpaca Dataset ---
# dataset = load_dataset("yahma/alpaca-cleaned", split="train")
# dataset = dataset.map(formatting_prompts_func, batched=True)

# --- Load & Preprocess the Alpaca Dataset ---
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Pre-tokenize the dataset: convert the "text" field to input_ids and attention_mask
def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=2048)

dataset = dataset.map(tokenize_batch, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# def my_data_collator(features):
#     return tokenizer.pad(
#         features,
#         padding=True,
#         pad_token_id=tokenizer.eos_token_id,
#         return_tensors="pt"
#     )
def my_data_collator(features):
    return tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt"
    )


# --- Set Up Training Configuration with Hugging Face SFTTrainer ---
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=20,
    max_steps=120,  # Adjust for full training runs
    learning_rate=5e-5,
    fp16=True,  # Use mixed precision if supported
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    # dataset_text_field="text",
    args=training_args,
    data_collator=my_data_collator,
)


# --- Fine-Tune the Model ---
trainer_stats = trainer.train()
print("Training complete. Stats:", trainer_stats)

# Save the fine-tuned adapter/model weights for later use
model.save_pretrained("qwen7b-finetuned-alpaca")
