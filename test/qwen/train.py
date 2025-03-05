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
    device_map="auto"
)

# Load the tokenizer; set pad_token if missing
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Apply LoRA Fine-Tuning ---
# Configure LoRA to adapt key projection layers (adjust target modules if needed)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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

EOS_TOKEN = tokenizer.eos_token  # Ensure EOS token is appended

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# --- Load & Preprocess the Alpaca Dataset ---
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

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
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,  # Adjust based on available CPU cores
    args=training_args,
)

# --- Fine-Tune the Model ---
trainer_stats = trainer.train()
print("Training complete. Stats:", trainer_stats)

# Save the fine-tuned adapter/model weights for later use
model.save_pretrained("qwen7b-finetuned-alpaca")
