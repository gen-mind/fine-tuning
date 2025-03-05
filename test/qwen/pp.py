import os
import gc
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
    TrainerCallback
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
        # Set the pad token
        tokenizer.pad_token = "<pad>"
    elif "<|pad|>" in tokenizer.get_vocab():
        print("<|pad|> token is in the tokenizer. Using <|pad|> for pad")
        # Set the pad token
        tokenizer.pad_token = "<|pad|>"
    elif "<unk>" in tokenizer.get_vocab():
        print("<unk> token is in the tokenizer. Using unk for pad")
        # Set the pad token
        tokenizer.pad_token = "<unk>"
    else:
        print(
            f"Using EOS token, {tokenizer.eos_token}, for padding. WARNING, this may not be ideal for chat fine-tuning models."
        )
        tokenizer.pad_token = tokenizer.eos_token

    # Update pad token id in model and its config
    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Check if they are equal
    assert model.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
    print("Tokenizer pad token ID:", tokenizer.pad_token_id)
    print("Model pad token ID:", model.pad_token_id)
    print("Model config pad token ID:", model.config.pad_token_id)
    print("Number of tokens now in tokenizer:", tokenizer.vocab_size)
    print("Special tokens map:", tokenizer.special_tokens_map)
    # print("All special tokens:", tokenizer.all_special_tokens)


def stream(model, user_prompt, model_type, tokenizer, checkpoint=""):
    """
    Streams text generation from the model based on the user prompt.
    """
    if model_type == "base":
        eval_model = model
    elif model_type == "fine-tuned":
        eval_model = PeftModel.from_pretrained(model, checkpoint)  # Assuming PeftModel is the intended class
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
        # copied from the test data set to ensure training is working
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
                f.flush()  # Force flush the buffered data to file

        # Check if the current step is a checkpoint step
        if state.global_step % int(args.save_steps) == 0:
            if state.best_model_checkpoint:
                checkpoint_dir = state.best_model_checkpoint
            else:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Save trainable params in the checkpoint directory
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

    # Check there are no parameters overflowing onto cpu (meta).
    for n, p in model.named_parameters():
        if p.device.type == "meta":
            print(f"{n} is on meta!")
    print("Max position embeddings:", model.config.max_position_embeddings)
    print("EOS token id:", model.config.eos_token_id)

    # Enable gradient checkpointing (comment this in to save on VRAM)
    model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model) # only set this if using quantization.

    # Set up PEFT with LoRA
    peft_config = LoraConfig(  # matching the Llama recipe
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "self_attn.rotary_emb.inv_freq",
            "gate_proj",
            "up_proj",
            "down_proj",
            # "lora_magnitude_vector", #required for DoRA
            # "input_layernorm.weight",
            # "post_attention_layernorm.weight",
            # "model.norm.weight",
            # "lm_head.weight",
            # "dense_h_to_4h", #for falcon
            # "dense_4h_to_h", #for falcon
            # "query_key_value", #for falcon
            # "dense" #for falcon
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        # use_dora=True # only for DoRA
    )
    model = get_peft_model(model, peft_config)  # move to a peft model

    print_trainable_parameters(model)

    # Set up Tokenizer and Padding
    print(tokenizer)
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    # Test the chat template
    messages = [
        {"role": "user", "content": "write a quick sort algorithm in python."},
        {"role": "assistant", "content": "here you are."},
        {"role": "user", "content": "great."},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Chat template output:", inputs)

    # Set up the pad token
    setup_pad_token(tokenizer, model)

    # Print the model generation config
    print("Generation Config:", model.generation_config)

    # ------------------------------
    # Load the Dataset
    # ------------------------------
    dataset = "Trelis/touch-rugby-rules-memorisation"
    data = load_dataset(dataset)
    print("First row of train:", data["train"][1])
    # print("First row of test:", data["test"][0])
    text = data["train"][0]["messages"]
    tokens = tokenizer.encode(text, add_special_tokens=True)
    decoded_text = tokenizer.decode(tokens)
    print("Token IDs:", tokens)
    print("Decoded Text:", decoded_text)

    # ------------------------------
    # Set up and run Training
    # ------------------------------
    global save_dir  # so that LoggingCallback can access it
    model_name = model_id.split("/")[-1]
    dataset_name = dataset.split("/")[-1]
    epochs = 1
    context_length = 512
    grad_accum = 1
    fine_tune_tag = "touch-rugby-rules"
    save_dir = f"./results/{model_name}_{dataset_name}_{epochs}_epochs_{context_length}_length-{fine_tune_tag}"
    print("Save directory:", save_dir)

    # Log file path for custom callback
    log_file_path = os.path.join(cache_dir, "training_logs.txt")
    logging_callback = LoggingCallback(log_file_path)

    # NOTE: Changed eval_steps from 0.2 to an integer value (50)
    trainer = SFTTrainer(
        # peft_config=peft_config, #comment out if passing a peft model directly as 'model'
        # dataset_text_field="messages",
        # max_seq_length=context_length,
        tokenizer=tokenizer,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        args=TrainingArguments(
            # max_steps=1, # comment this out after the first time you run. This is for testing!
            save_steps=50,  ### MAKE SURE TO CHECK THIS VALUE IS GOOD FOR YOUR RUN!
            logging_steps=1,
            num_train_epochs=epochs,
            output_dir=save_dir,
            evaluation_strategy="steps",
            do_eval=True,
            eval_steps=50,  # Changed from 0.2 (float) to 50 (int)
            per_device_eval_batch_size=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            log_level="debug",
            # optim="paged_adamw_8bit",
            # fp16=True, #For non-Ampere GPUs
            bf16=True,  # For Ampere GPUs
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            hub_private_repo=True,
            warmup_ratio=0.03,  # optional, may help stability at the start of training. Not required for simple fine-tunes.
            optim="adamw_torch",  # comment out for LoRA +
            learning_rate=1e-4,  # comment out for LoRA +
            remove_unused_columns=False,
        ),
        callbacks=[logging_callback],
        # optimizers=(optimizer, None),  # Comment in for LoRA+
        # neftune_noise_alpha=5 # Add in noise to embeddings to improve performance!
    )

    # ------------------------------
    # Before training, silence cache warnings
    # ------------------------------
    model.config.use_cache = False  # Fixed typo: changed "odel.config.use_cache" to "model.config.use_cache"

    # Start Training
    trainer.train()

    # Plot training and evaluation losses
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

    # Run evaluation on the base model
    evaluation(model, "base", tokenizer)

    # ------------------------------
    # Save and Push the Model to Hub
    # ------------------------------
    adapter_model = f"Trelis/{model_name}-{fine_tune_tag}-adapters"
    new_model = f"Trelis/{model_name}-{fine_tune_tag}"  # adjust 'Trelis' to your HuggingFace organisation

    # Save the model with adapters locally and push to hub
    model.save_pretrained(f"{model_name}-{fine_tune_tag}-adapters-local", push_to_hub=True, use_auth_token=True)
    model.push_to_hub(adapter_model, use_auth_token=True, max_shard_size="10GB", use_safetensors=True)

    # Upload the trainable_params as well
    from huggingface_hub import HfApi, create_repo, create_branch

    create_repo(new_model, private=True)
    create_branch(new_model, repo_type="model", branch="gguf")

    # Initialize the HfApi class
    api = HfApi()

    repo_id = adapter_model
    local_file_paths = [
        os.path.join(save_dir, "trainable_params_final.bin"),
    ]
    for local_file_path in local_file_paths:
        file_name = local_file_path.split("/")[-1]
        path_in_repo = file_name  # Using file_name directly, adjust as needed
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",  # Assuming it's a model; can be "dataset" or "space" as well
        )
        print(f"Uploaded {file_name} to {repo_id}")

    model = model.merge_and_unload()

    model.save_pretrained(f"{model_name}-{fine_tune_tag}-local")
    tokenizer.save_pretrained(f"{model_name}-{fine_tune_tag}-local")


if __name__ == "__main__":
    main()
