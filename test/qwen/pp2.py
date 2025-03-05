import os
import gc
import json
import torch
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
    TextStreamer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

# ------------------------------
# Global configuration variables
# ------------------------------
model_id = "openchat/openchat_3.5"
# model_id = "Qwen/Qwen1.5-7B-Chat"
cache_dir = "cache"


# ------------------------------
# Utility Functions
# ------------------------------
def load_model_and_tokenizer(model_id, cache_dir):
    # BitsAndBytes configuration for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Optionally load custom configuration
    # config = AutoConfig.from_pretrained(model_id)
    # config.max_position_embeddings = 4096 # (input + output) tokens can now be up to 4096

    # Load the model with specified parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # config=config,
        # quantization_config=bnb_config,
        # rope_scaling={"type": "linear", "factor": 2.0},
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # works with Llama models and reduces memory reqs
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )
    return model, tokenizer


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model and lists which parameters are trainable.
    """
    trainable_params = 0
    non_trainable_params = 0
    all_params = 0

    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  {name}")
        else:
            non_trainable_params += param.numel()

    print("\nNon-Trainable Parameters:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  {name}")

    print(
        f"\nSummary:\n  Trainable params: {trainable_params}\n  Non-Trainable params: {non_trainable_params}\n  All params: {all_params}\n  Trainable%: {100 * trainable_params / all_params}"
    )


def setup_pad_token(tokenizer, model):
    """
    Set the pad token based on availability in the tokenizer.
    """
    ## OPTION A - set the pad token to <pad>, if not <|pad|>, if not <unk> if <unk> is in the tokenizer OR set it to the EOS token.
    if "<pad>" in tokenizer.get_vocab():
        print("<pad> token is in the tokenizer. Using <pad> for pad")
        tokenizer.pad_token = "<pad>"
    elif "<|pad|>" in tokenizer.get_vocab():
        print("<|pad|> token is in the tokenizer. Using <|pad|> for pad")
        tokenizer.pad_token = "<|pad|>"
    elif "<unk>" in tokenizer.get_vocab():
        print("<unk> token is in the tokenizer. Using unk for pad")
        tokenizer.pad_token = "<unk>"
    else:
        print(
            f"Using EOS token, {tokenizer.eos_token}, for padding. WARNING, this may not be ideal for chat fine-tuning models."
        )
        tokenizer.pad_token = tokenizer.eos_token

    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    assert model.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
    print("Tokenizer pad token ID:", tokenizer.pad_token_id)
    print("Model pad token ID:", model.pad_token_id)
    print("Model config pad token ID:", model.config.pad_token_id)
    print("Number of tokens now in tokenizer:", tokenizer.vocab_size)
    print("Special tokens map:", tokenizer.special_tokens_map)


def preprocess_sample(example):
    """
    Converts the 'messages' field (a JSON string) into a plain-text conversation stored in 'text'.
    """
    try:
        messages_json = json.loads(example["messages"])["messages"]
    except Exception as e:
        print("Error parsing JSON:", e)
        messages_json = []
    conversation = ""
    for msg in messages_json:
        conversation += f"{msg['role']}: {msg['content']}\n"
    example["text"] = conversation.strip()


    return example

def tokenize_function(example, tokenizer, context_length=512):
    # Tokenize the 'text' field with padding and truncation.
    output = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=context_length
    )
    # For causal language modeling, labels are the same as input_ids.
    output["labels"] = output["input_ids"].copy()
    return output





def stream(model, user_prompt, model_type, tokenizer, checkpoint=""):
    """
    Streams text generation from the model based on the user prompt.
    """
    if model_type == "base":
        eval_model = model
    elif model_type == "fine-tuned":
        eval_model = PeftModel.from_pretrained(model, checkpoint)
        eval_model = eval_model.to("cuda")
        for n, p in eval_model.named_parameters():
            if p.device.type == "cpu":
                print(f"{n} is on cpu!")
    else:
        print("You must set the model_type to base or fine-tuned")
        exit()

    eval_model.config.use_cache = True

    messages = [{"role": "user", "content": user_prompt.strip()}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([inputs], return_tensors="pt", add_special_tokens=False).to("cuda")

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    streamer = TextStreamer(tokenizer)
    print(f"eval_model is on: {next(eval_model.parameters()).device}")
    print(f"input_ids are on: {inputs['input_ids'].device}")

    _ = eval_model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    torch.cuda.empty_cache()
    gc.collect()


def evaluation(model, model_type, tokenizer, checkpoint=""):
    """
    Evaluates the model on a set of predefined questions.
    """
    questions = [
        "In the context of Touch Rugby International Playing Rules 2020, what is the purpose of the Dead Ball Line?",
        "How many players are on the field on each team in touch rugby?",
        "In touch rugby, does a forward pass result in a roll ball, a scrum, or something else?",
        "In touch rugby, how many metres must the defending team retreat after a touch?",
        "In touch rugby, how many substitutions are allowed during a game?",
        "In touch rugby, how long is half time?",
        "In touch rugby, how does the game commence?",
        "In touch rugby, how many metres must defenders retreat when there is a penalty? Is the same as after a touch is made?",
        "In touch rugby, how many touches is a team entitled to prior to a change in possession?",
        "In touch rugby, what happens if a player makes a pass after a touch has been made?",
        "In touch rugby, how many points is a try worth?",
    ]

    answers = [
        "The Dead Ball Line marks the end boundaries of the field of play and indicates when the ball is out of play.",
        "6 players.",
        "Penalty.",
        "7 metres.",
        "There is no limit.",
        "5 minutes.",
        "The game begins with a tap on the halfway line.",
        "10 metres.",
        "Possession changes on the sixth (6th) touch.",
        "The defending team gains possession and a penalty.",
        "1 point.",
    ]

    for question, answer in zip(questions, answers):
        stream(model, question, model_type, tokenizer, checkpoint)
        print("Correct Answer:", answer)
        print('\n\n')


# ------------------------------
# Custom Callback for Logging
# ------------------------------
class LoggingCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.save_dir = save_dir

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        with open(self.log_file_path, "a") as f:
            if logs is not None:
                if "loss" in logs:
                    f.write(f"Step: {state.global_step}, Training Loss: {logs['loss']}\n")
                if "eval_loss" in logs:
                    f.write(f"Step: {state.global_step}, Eval Loss: {logs['eval_loss']}\n")
                f.flush()

        if state.global_step % int(args.save_steps) == 0:
            if state.best_model_checkpoint:
                checkpoint_dir = state.best_model_checkpoint
            else:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            current_trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            current_trainable_params_state_dict = {n: p.data for n, p in current_trainable_params.items()}
            file_path = os.path.join(checkpoint_dir, "trainable_params.bin")
            torch.save(current_trainable_params_state_dict, file_path)


# ------------------------------
# Main Training and Execution
# ------------------------------
def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id, cache_dir)

    for n, p in model.named_parameters():
        if p.device.type == "meta":
            print(f"{n} is on meta!")
    print("Max position embeddings:", model.config.max_position_embeddings)
    print("EOS token id:", model.config.eos_token_id)

    model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model) # only set this if using quantization.

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)

    print(tokenizer)
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    messages = [
        {"role": "user", "content": "write a quick sort algorithm in python."},
        {"role": "assistant", "content": "here you are."},
        {"role": "user", "content": "great."},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Chat template output:", inputs)

    setup_pad_token(tokenizer, model)
    print("Generation Config:", model.generation_config)

    # ------------------------------
    # Load and Preprocess the Dataset
    # ------------------------------
    dataset = "Trelis/touch-rugby-rules-memorisation"
    data = load_dataset(dataset)
    # Preprocess both train and test splits to add a "text" field
    data["train"] = data["train"].map(preprocess_sample)
    if "test" in data:
        data["test"] = data["test"].map(preprocess_sample)
    else:
        # If no test split, use a copy of train for evaluation (not ideal, but for example purposes)
        data["test"] = data["train"]

    # Then, when calling map, do something like:
    tokenized_train = data["train"].map(lambda x: tokenize_function(x, tokenizer), batched=False)
    tokenized_eval = data["test"].map(lambda x: tokenize_function(x, tokenizer), batched=False)


    print("Number of train samples:", len(data["train"]))
    print("Number of test samples:", len(data["test"]))

    # For inspection, print first row text and tokenize it.
    print("First row of train:", data["train"][1])
    sample_text = data["train"][0]["text"]
    tokens = tokenizer.encode(sample_text, add_special_tokens=True)
    decoded_text = tokenizer.decode(tokens)
    print("Token IDs:", tokens)
    print("Decoded Text:", decoded_text)

    # ------------------------------
    # Set up and run Training
    # ------------------------------
    global save_dir
    model_name = model_id.split("/")[-1]
    dataset_name = dataset.split("/")[-1]
    epochs = 1
    context_length = 512
    grad_accum = 1
    fine_tune_tag = "touch-rugby-rules"
    save_dir = f"./results/{model_name}_{dataset_name}_{epochs}_epochs_{context_length}_length-{fine_tune_tag}"
    print("Save directory:", save_dir)

    log_file_path = os.path.join(cache_dir, "training_logs.txt")
    logging_callback = LoggingCallback(log_file_path)

    trainer = SFTTrainer(
        processing_class=tokenizer,  # Use processing_class instead of the deprecated tokenizer argument
        model=model,
        # train_dataset=data["train"],
        # eval_dataset=data["test"],
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        args=TrainingArguments(
            save_steps=50,
            logging_steps=1,
            num_train_epochs=epochs,
            output_dir=save_dir,
            evaluation_strategy="steps",
            do_eval=True,
            eval_steps=50,
            per_device_eval_batch_size=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            log_level="debug",
            bf16=True,
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            hub_private_repo=True,
            warmup_ratio=0.03,
            optim="adamw_torch",
            learning_rate=1e-4,
            remove_unused_columns=False,
        ),
        callbacks=[logging_callback],
    )

    model.config.use_cache = False

    trainer.train()

    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            train_losses.append(entry["loss"])
            train_steps.append(entry["step"])
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
            eval_steps.append(entry["step"])
    plt.figure()
    plt.plot(train_steps, train_losses, label="Train Loss")
    plt.plot(eval_steps, eval_losses, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    evaluation(model, "base", tokenizer)

    adapter_model = f"Trelis/{model_name}-{fine_tune_tag}-adapters"
    new_model = f"Trelis/{model_name}-{fine_tune_tag}"

    model.save_pretrained(f"{model_name}-{fine_tune_tag}-adapters-local", push_to_hub=True, use_auth_token=True)
    model.push_to_hub(adapter_model, use_auth_token=True, max_shard_size="10GB", use_safetensors=True)

    from huggingface_hub import HfApi, create_repo, create_branch

    create_repo(new_model, private=True)
    create_branch(new_model, repo_type="model", branch="gguf")

    api = HfApi()
    repo_id = adapter_model
    local_file_paths = [os.path.join(save_dir, "trainable_params_final.bin")]
    for local_file_path in local_file_paths:
        file_name = os.path.basename(local_file_path)
        path_in_repo = file_name
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Uploaded {file_name} to {repo_id}")

    model = model.merge_and_unload()
    model.save_pretrained(f"{model_name}-{fine_tune_tag}-local")
    tokenizer.save_pretrained(f"{model_name}-{fine_tune_tag}-local")


if __name__ == "__main__":
    main()
